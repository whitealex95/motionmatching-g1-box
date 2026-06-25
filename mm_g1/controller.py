"""Real-time motion matching with a pick / carry / place box skill on top of GenoView loco.

The locomotion core is a faithful port of genoview_g1.py (a Savitzky-Golay-smoothed
"simulation root" the matcher tracks + integrates, per-clip KD-tree search, inertialized
cuts); see git history for the unchanged details. Layered on top is a small box-manipulation
state machine driven by the B trigger:

    LOCOMOTION --B (near box)--> PICK (ride) --> CARRY (search) --B--> PLACE (ride) --> LOCOMOTION

PICK and PLACE are *ridden* like the jump (no search mid-skill, entered from their start by a
nearest-neighbour match of the live pose + box pose). CARRY is searched every SEARCH_TIME like
locomotion, but only among CARRY frames and with the box pose added to the query. The only
database transitions ever made are exactly those in the chain above (req. 13):
  loco->loco, loco->pick, pick->carry, carry->carry, carry->place, place->loco.

Each searchable database has its own feature space (features.build_db):
  loco  (27)  pose + future trajectory                         -- unchanged genoview features
  carry (36)  pose + future trajectory + box(pos,ori,vel)      -- box added to the query
  pick  (24)  pose + box(pos,ori,vel)                           -- NO trajectory; box pos weighted
  place (24)  pose + box(pos,ori,vel)                           -- NO trajectory
pick and place share the 24-D layout but are separate databases, so pick can weight the box
position more heavily (PICK_BOX_POS_WEIGHT) -- the entry is chosen mainly by where the box is.

The box rides the robot's gravity-aligned base frame while held (stored per frame as
boxLocal{Pos,Rot}); before pick contact and after place release it rests in the world.
"""
import numpy as np
from scipy.spatial import cKDTree

from . import config as C
from . import quat
from . import boxes
from .features import build_db, yaw_quat, FORWARD, HORIZONS, FPS
from .jumps import jump_entries
from .springs import (DecaySpringDamperPosition, DecaySpringDamperRotation,
                      TrajectorySpringPosition, TrajectorySpringRotation)

DT = C.DT
NDOF = 29
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


class MotionMatcher:
    def __init__(self, lib, start_frame=None):
        self.lib = lib
        db = self.db = build_db(lib)
        self.starts, self.stops = db["starts"], db["stops"]
        self.dof, self.dofVel = db["dof"], db["dofVel"]
        self.simPosDB, self.simThetaDB = db["simPos"], db["simTheta"]
        self.simVelDB, self.yawRateDB = db["simVel"], db["yawRate"]
        self.plpDB, self.plvDB = db["pelvLocalPos"], db["pelvLocalVel"]
        self.prDB, self.paDB = db["pelvLocalRot"], db["pelvLocalAng"]
        self.boxLocalPos, self.boxLocalRot = db["boxLocalPos"], db["boxLocalRot"]
        self.boxLocalPosVel, self.boxLocalAng = db["boxLocalPosVel"], db["boxLocalAng"]
        # Per-skill normalized feature matrices + raw blocks for cross-database queries.
        self.Xloco, self.Xcarry = db["dbs"]["loco"]["X"], db["dbs"]["carry"]["X"]
        self.X = self.Xloco                                  # back-compat (jump entry uses it)
        self.rawXpos, self.rawXvel = db["rawXpos"], db["rawXvel"]
        self.clip_id = lib["clip_id"]
        self.skill = lib["skill"]
        self.box_attach = lib["box_attach"]
        self.Ttimes = HORIZONS / FPS
        TAIL = HORIZONS[-1]

        # ---- Locomotion KD-trees: one per locomotion clip (skill==0 everywhere) ----
        self.loco_trees = []                                 # (rs, re, tree)
        for rs, re in zip(self.starts, self.stops):
            if self.skill[rs:re].any() or re - rs <= TAIL:   # skip jump/box clips & tiny clips
                continue
            self.loco_trees.append((int(rs), int(re), cKDTree(self.Xloco[rs:re - TAIL])))

        # ---- Carry KD-trees: one per contiguous CARRY segment (searched like locomotion) ----
        self.carry_trees = []                                # (rs, re, tree)
        for rs, re in boxes.carry_segments(lib):
            if re - rs <= TAIL:
                continue
            self.carry_trees.append((rs, re, cKDTree(self.Xcarry[rs:re - TAIL])))

        # ---- Pick / place ENTRY frames (nearest-neighbour matched, then ridden) ----
        self.pick_enter, self.pick_end_of, self.place_enter, self.place_end_of = \
            boxes.box_entries(lib)

        # Legacy J jump entries (unchanged).
        self.jump_enter, self.jump_land_of = jump_entries(lib)

        # Box resting orientation for the interactive spawn: the box pose in the base frame at
        # a representative pick entry == its world pose when the robot faces +x (yaw 0).
        self.box_spawn_rot = (self.boxLocalRot[self.pick_enter[0]].copy()
                              if len(self.pick_enter) else IDENTITY.copy())
        self.reset(start_frame)

    # --- state ---------------------------------------------------------------
    def reset(self, start_frame=None):
        if start_frame is None:
            start_frame = min(self.stops[0] - 1, self.starts[0] + 30)
        self.lo, self.hi = self._clip_bounds(start_frame)
        self.animFrame = int(start_frame)
        self.state = C.SKILL_LOCO
        # Controller root = the smoothed simulation root (ground position + yaw).
        self.rootPos = self.simPosDB[self.animFrame].copy()
        self.rootVel = np.zeros(3); self.rootAcc = np.zeros(3); self.rootAng = np.zeros(3)
        self.rootYaw = float(self.simThetaDB[self.animFrame])
        self.rootRot = yaw_quat(self.rootYaw)
        self.desiredDir = quat.mul_vec(self.rootRot, FORWARD)
        # Inertialization offsets: joints, pelvis-local position, pelvis-local rotation.
        self.offDof = np.zeros(NDOF); self.offDofVel = np.zeros(NDOF)
        self.offPP = np.zeros(3); self.offPPVel = np.zeros(3)
        self.offPR = IDENTITY.copy(); self.offPAng = np.zeros(3)
        self.searchTimer = 0.0
        self.jump_pending = False
        self.jump_locked = 0
        self.box_pending = False
        self.box_locked = 0
        # Box world pose: a reachable distance in front of the robot's start facing, at its
        # resting height, oriented as it rests in the data (box_spawn_rot is in the base frame,
        # so composing with the start root gives the matching world orientation).
        self.boxPos = self.rootPos + quat.mul_vec(
            self.rootRot, np.array([C.BOX_SPAWN_FWD, C.BOX_SPAWN_LAT, 0.0]))
        self.boxPos[2] = C.BOX_REST_Z
        self.boxRot = quat.mul(self.rootRot, self.box_spawn_rot)
        self.boxVelWorld = np.zeros(3); self.boxPosPrev = self.boxPos.copy()
        self.box_held = False
        self.offBoxP = np.zeros(3); self.offBoxPVel = np.zeros(3)
        self.offBoxR = IDENTITY.copy(); self.offBoxAng = np.zeros(3)
        self.Tpos = np.tile(self.rootPos, (len(HORIZONS), 1))   # command-trajectory viz
        self.Tdir = np.tile(self.desiredDir, (len(HORIZONS), 1))

    def _clip_bounds(self, frame):
        """[lo, hi) playable range of the clip containing `frame`."""
        r = int(np.searchsorted(self.starts, frame, "right") - 1)
        return int(self.starts[r]), int(self.stops[r])

    @property
    def cur(self):
        return self.animFrame

    @property
    def jumping(self):
        return self.jump_locked > 0

    @property
    def near_box(self):
        """True when the root is within PICK_RADIUS of the box (planar)."""
        return float(np.linalg.norm(self.rootPos[:2] - self.boxPos[:2])) < C.PICK_RADIUS

    def state_name(self):
        return {C.SKILL_LOCO: "LOCOMOTION", C.SKILL_PICK: "PICK",
                C.SKILL_CARRY: "CARRY", C.SKILL_PLACE: "PLACE"}[self.state]

    # --- triggers ------------------------------------------------------------
    def trigger_jump(self):
        """Request a jump (J). Honoured next step if idle (not jumping / not handling a box)."""
        if self.jump_locked == 0 and self.box_locked == 0 and self.state == C.SKILL_LOCO:
            self.jump_pending = True

    def trigger_box(self):
        """Request the box action (B): pick up if near a box in locomotion, or place if
        carrying. Honoured on the next step; a no-op while a skill is already being ridden."""
        if self.box_locked == 0 and self.jump_locked == 0:
            self.box_pending = True

    # --- inertialized cut ----------------------------------------------------
    def _inertialize_into(self, b, lo, hi):
        """Capture the pose discontinuity from the current frame to frame `b` (joints,
        pelvis-local position + rotation) as decaying inertialization offsets, then move the
        playhead there with new playable bounds [lo, hi) -- no pop."""
        a = self.animFrame
        self.offDof = (self.offDof + self.dof[a]) - self.dof[b]
        self.offDofVel = (self.offDofVel + self.dofVel[a]) - self.dofVel[b]
        self.offPP = (self.offPP + self.plpDB[a]) - self.plpDB[b]
        self.offPPVel = (self.offPPVel + self.plvDB[a]) - self.plvDB[b]
        self.offPR = quat.abs(quat.mul_inv(quat.mul(self.offPR, self.prDB[a]), self.prDB[b]))
        self.offPAng = (self.offPAng + self.paDB[a]) - self.paDB[b]
        # The held box gets the same treatment: capture the box-in-base discontinuity from a
        # to b (position + rotation, with their local velocity terms) so it never pops at a
        # carry search cut or a pick->carry / carry->place hand-off. When the box is resting
        # (not held) there is nothing to carry across -- the grab in _update_box re-seeds it.
        if self.box_held:
            self.offBoxP = (self.offBoxP + self.boxLocalPos[a]) - self.boxLocalPos[b]
            self.offBoxPVel = (self.offBoxPVel + self.boxLocalPosVel[a]) - self.boxLocalPosVel[b]
            self.offBoxR = quat.abs(quat.mul_inv(
                quat.mul(self.offBoxR, self.boxLocalRot[a]), self.boxLocalRot[b]))
            self.offBoxAng = (self.offBoxAng + self.boxLocalAng[a]) - self.boxLocalAng[b]
        self.animFrame, self.lo, self.hi = int(b), int(lo), int(hi)

    # --- one real-time frame -------------------------------------------------
    def step(self, desiredVel, desiredFace):
        """Advance one frame. desiredVel is the desired velocity [x,y,0] (WASD), desiredFace an
        independent facing [x,y,0] (arrows; zero = face travel). Returns world qpos (36,); the
        box world pose is exposed as self.boxPos / self.boxRot for the viewer to draw."""
        desiredVel = np.asarray(desiredVel, float)
        self._predict_trajectory(desiredVel, desiredFace)
        self._maybe_enter_jump()
        self._maybe_trigger_box()
        return self._query_from_trajectory(desiredVel)

    def _maybe_trigger_box(self):
        """Honour a pending B: locomotion + near box -> enter PICK; carry -> enter PLACE."""
        if not self.box_pending or self.box_locked > 0 or self.jump_locked > 0:
            self.box_pending = False
            return
        self.box_pending = False
        if self.state == C.SKILL_LOCO and self.near_box:
            self._enter_skill(self.pick_enter, self.pick_end_of, C.SKILL_PICK, "pick")
        elif self.state == C.SKILL_CARRY:
            self._enter_skill(self.place_enter, self.place_end_of, C.SKILL_PLACE, "place")

    def _enter_skill(self, enter_frames, end_of, skill, dbname):
        """Nearest-neighbour match the live pose + box pose to the start of a pick/place phase
        (in that skill's own database `dbname`), inertialize into it, and lock the skill so it
        is ridden to the phase end."""
        if len(enter_frames) == 0:
            return
        Xq = self._query(dbname)
        Xdb = self.db["dbs"][dbname]["X"]
        entry = int(enter_frames[np.argmin(np.linalg.norm(Xdb[enter_frames] - Xq, axis=1))])
        end = int(end_of[entry])
        lo, _ = self._clip_bounds(entry)
        self._inertialize_into(entry, lo, end + 1)           # hi caps the ride at the phase end
        self.state = skill
        self.box_locked = max(1, end - entry)
        self.searchTimer = C.SEARCH_TIME

    def _finish_ride(self):
        """Called when a PICK/PLACE ride ends: transition into the next searchable state."""
        if self.state == C.SKILL_PICK:                       # pick -> carry
            res = self._best_carry()
            if res is not None:
                f, lo, hi = res
                self._inertialize_into(f, lo, hi)
                self.state = C.SKILL_CARRY
        else:                                                # place -> locomotion
            f, lo, hi = self._best_loco()
            self._inertialize_into(f, lo, hi)
            self.state = C.SKILL_LOCO
        self.searchTimer = C.SEARCH_TIME

    # --- jump skill (legacy J) -----------------------------------------------
    def _best_jump_entry(self):
        if len(self.jump_enter) == 0:
            return None
        d = np.linalg.norm(self.X[self.jump_enter] - self.X[self.animFrame], axis=1)
        return int(self.jump_enter[d.argmin()])

    def _maybe_enter_jump(self):
        if self.jump_pending and self.jump_locked == 0 and self.box_locked == 0:
            self.jump_pending = False
            entry = self._best_jump_entry()
            if entry is not None:
                lo, hi = self._clip_bounds(entry)
                self._inertialize_into(entry, lo, hi)
                land = self.jump_land_of[entry]
                after_end = min(land + 1 + C.PHASE_TOUCHDOWN + C.PHASE_AFTER, hi - 1)
                self.jump_locked = max(1, after_end - entry)
                self.searchTimer = C.SEARCH_TIME

    # --- predict the desired trajectory (query) ------------------------------
    def _predict_trajectory(self, desiredVel, desiredFace):
        desiredVel = np.asarray(desiredVel, float)
        if np.linalg.norm(desiredFace) > 0.01:
            self.desiredDir = np.asarray(desiredFace, float) / np.linalg.norm(desiredFace)
        elif np.linalg.norm(desiredVel) > 0.01:
            self.desiredDir = desiredVel / np.linalg.norm(desiredVel)
        desiredRot = yaw_quat(np.arctan2(self.desiredDir[1], self.desiredDir[0]))
        dt_col = self.Ttimes[:, None]
        self.Tpos, _, _ = TrajectorySpringPosition(
            self.rootPos, self.rootVel, self.rootAcc, desiredVel, C.VEL_HALFLIFE, dt_col)
        Trot, _ = TrajectorySpringRotation(
            self.rootRot, self.rootAng, desiredRot, C.ROT_HALFLIFE, dt_col)
        self.Tdir = quat.mul_vec(Trot, FORWARD)

    # --- query assembly (one normalized vector per database) -----------------
    def _box_local_live(self, qh):
        """Live box pose in the controller base frame (yaw qh at the ground root)."""
        bp = quat.inv_mul_vec(qh, self.boxPos - self.rootPos)
        br = quat.mul(quat.inv(qh), self.boxRot)
        baa = quat.to_scaled_angle_axis(quat.abs(br))
        bv = quat.inv_mul_vec(qh, self.boxVelWorld)
        return bp, baa, bv

    def _query(self, name):
        """Assemble + normalize the search query for database `name`
        ('loco'/'carry'/'pick'/'place'). Pose blocks come from the current frame, trajectory
        from the command springs, box from the live box -- in the exact block order
        features.build_db concatenated them (trajectory only for loco/carry; box for
        carry/pick/place)."""
        d = self.db["dbs"][name]
        qh = yaw_quat(self.rootYaw)
        parts = [self.rawXpos[self.animFrame], self.rawXvel[self.animFrame]]
        if name in ("loco", "carry"):
            parts.append(quat.inv_mul_vec(qh, self.Tpos - self.rootPos)[:, 0:2].ravel())
            parts.append(quat.inv_mul_vec(qh, self.Tdir)[:, 0:2].ravel())
        if name in ("carry", "pick", "place"):
            parts.extend(self._box_local_live(qh))
        q = np.concatenate(parts)
        return (q - d["offset"]) / d["scale"]

    # --- searches over each database -----------------------------------------
    def _search_trees(self, trees, Xq, Xself, with_bias):
        """Generic per-segment KD-tree search. Returns (frame, lo, hi) -- the current frame &
        bounds if nothing beats it. `Xself` is the database matrix the current-frame bias is
        measured in (so the stay-in-place bias uses the same feature space as the trees)."""
        bestF, bestLo, bestHi = self.animFrame, self.lo, self.hi
        if with_bias and self.animFrame < self.hi - HORIZONS[-1]:
            best = float(np.linalg.norm(Xq - Xself[self.animFrame]) - C.CURRENT_BIAS)
        else:
            best = np.inf
        for rs, re, tree in trees:
            dist, k = tree.query(Xq, eps=C.APPROX_BIAS, distance_upper_bound=best)
            if dist < best:
                best, bestF, bestLo, bestHi = dist, int(rs + k), rs, re
        return bestF, bestLo, bestHi

    def _search_loco(self):
        f, lo, hi = self._search_trees(self.loco_trees, self._query("loco"), self.Xloco, True)
        if f != self.animFrame:
            self._inertialize_into(f, lo, hi)
        else:
            self.lo, self.hi = lo, hi

    def _search_carry(self):
        f, lo, hi = self._search_trees(self.carry_trees, self._query("carry"), self.Xcarry, True)
        if f != self.animFrame:
            self._inertialize_into(f, lo, hi)
        else:
            self.lo, self.hi = lo, hi

    def _best_carry(self):
        """Best CARRY frame for the pick->carry hand-off (no stay-in-place bias)."""
        if not self.carry_trees:
            return None
        return self._search_trees(self.carry_trees, self._query("carry"), self.Xcarry, False)

    def _best_loco(self):
        """Best locomotion frame for the place->locomotion hand-off."""
        return self._search_trees(self.loco_trees, self._query("loco"), self.Xloco, False)

    # --- match + advance + reconstruct ---------------------------------------
    def _query_from_trajectory(self, desiredVel=None):
        # ---- Search (skipped while riding a jump or a pick/place skill) ----
        if self.jump_locked == 0 and self.box_locked == 0 and self.searchTimer <= 0.0:
            if self.state == C.SKILL_CARRY:
                self._search_carry()
            else:
                self._search_loco()
            self.searchTimer = C.SEARCH_TIME

        # ---- Advance the playhead within its current bounds ----
        self.animFrame = int(np.clip(self.animFrame + 1, self.lo, self.hi - 1))
        self.searchTimer -= DT
        if self.jump_locked > 0:
            self.jump_locked -= 1
            if self.jump_locked == 0:
                self.searchTimer = 0.0
        elif self.box_locked > 0:
            self.box_locked -= 1
            if self.box_locked == 0:
                self._finish_ride()
        elif self.animFrame >= self.hi - 2:
            self.searchTimer = 0.0
        f = self.animFrame

        # ---- Integrate controller root from the matched clip's smooth root velocity ----
        if desiredVel is not None:
            _, _, self.rootAcc = TrajectorySpringPosition(
                self.rootPos, self.rootVel, self.rootAcc, desiredVel, C.ROT_HALFLIFE, DT)
        qh_clip = yaw_quat(self.simThetaDB[f])
        clipVelLocal = quat.inv_mul_vec(qh_clip, self.simVelDB[f])
        self.rootVel = quat.mul_vec(self.rootRot, clipVelLocal)
        self.rootAng = np.array([0.0, 0.0, self.yawRateDB[f]])
        self.rootPos = self.rootPos + self.rootVel * DT
        self.rootYaw = self.rootYaw + self.yawRateDB[f] * DT
        self.rootRot = yaw_quat(self.rootYaw)

        # ---- Inertialize joints + pelvis-local offset, then reconstruct the pose ----
        self.offDof, self.offDofVel = DecaySpringDamperPosition(
            self.offDof, self.offDofVel, C.INERT_HALFLIFE, DT)
        self.offPP, self.offPPVel = DecaySpringDamperPosition(
            self.offPP, self.offPPVel, C.INERT_HALFLIFE, DT)
        self.offPR, self.offPAng = DecaySpringDamperRotation(
            self.offPR, self.offPAng, C.INERT_HALFLIFE, DT)

        dofOut = self.dof[f] + self.offDof
        pelvLocalPos = self.plpDB[f] + self.offPP
        pelvLocalRot = quat.mul(self.offPR, self.prDB[f])
        pelvWorldPos = self.rootPos + quat.mul_vec(self.rootRot, pelvLocalPos)
        pelvWorldRot = quat.mul(self.rootRot, pelvLocalRot)

        # ---- Reconstruct the box (rides the root while held; frozen otherwise) ----
        self._update_box(f)

        qpos = np.empty(36)
        qpos[0:3] = pelvWorldPos
        qpos[3:7] = pelvWorldRot
        qpos[7:] = dofOut
        return qpos

    def _update_box(self, f):
        """Place the box. While the matched frame is `attached`, the box rides the controller
        root as `root o (boxLocal + inertialization offset)` -- the exact pelvis construction,
        so the box pose is C1-continuous through every cut (the offsets are accumulated in
        _inertialize_into and decayed here). At the grab the resting->held discontinuity is
        captured as a base-local offset too, so the box eases into the hands rather than
        snapping. When not attached the box stays put -- on the floor before pick contact, and
        wherever it was set down after place release."""
        attached = (self.state in (C.SKILL_PICK, C.SKILL_CARRY, C.SKILL_PLACE)
                    and bool(self.box_attach[f]))
        if attached:
            if not self.box_held:                            # grab: seed the base-local offset
                localP = quat.inv_mul_vec(self.rootRot, self.boxPos - self.rootPos)
                localR = quat.mul(quat.inv(self.rootRot), self.boxRot)
                self.offBoxP = localP - self.boxLocalPos[f]
                self.offBoxPVel = np.zeros(3)
                self.offBoxR = quat.abs(quat.mul_inv(localR, self.boxLocalRot[f]))
                self.offBoxAng = np.zeros(3)
                self.box_held = True
            self.offBoxP, self.offBoxPVel = DecaySpringDamperPosition(
                self.offBoxP, self.offBoxPVel, C.BOX_INERT_HALFLIFE, DT)
            self.offBoxR, self.offBoxAng = DecaySpringDamperRotation(
                self.offBoxR, self.offBoxAng, C.BOX_INERT_HALFLIFE, DT)
            boxLocalPos = self.boxLocalPos[f] + self.offBoxP
            boxLocalRot = quat.mul(self.offBoxR, self.boxLocalRot[f])
            self.boxPos = self.rootPos + quat.mul_vec(self.rootRot, boxLocalPos)
            self.boxRot = quat.mul(self.rootRot, boxLocalRot)
        else:
            self.box_held = False                            # frozen at its current world pose
        self.boxVelWorld = (self.boxPos - self.boxPosPrev) / DT
        self.boxPosPrev = self.boxPos.copy()

    def box_qpos(self):
        """Box freejoint qpos (7,) = world position + quaternion (wxyz), for the scene."""
        return np.concatenate([self.boxPos, self.boxRot])
