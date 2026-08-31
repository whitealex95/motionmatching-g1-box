"""Real-time motion matching with a pick / carry / place box skill on top of GenoView loco.

The locomotion core is a faithful port of genoview_g1.py (a Savitzky-Golay-smoothed
"simulation root" the matcher tracks + integrates, per-clip KD-tree search, inertialized
cuts); see git history for the unchanged details. Layered on top is a small box-manipulation
state machine driven by the B trigger:

    LOCOMOTION --B (near box)--> PICK (ride) --> CARRY (search) --B--> PLACE (ride) --> LOCOMOTION

PICK and PLACE are *ridden* (no search mid-phase, entered from their start by a
nearest-neighbour match of the live pose + box pose). CARRY is searched every SEARCH_TIME like
locomotion, but only among CARRY frames and with the box pose added to the query. The only
database transitions ever made are exactly those in the chain above (req. 13):
  loco->loco, loco->pick, pick->carry, carry->carry, carry->place, place->loco.

Each searchable database has its own feature space (features.build_db):
  loco  (27)  pose + future trajectory                         -- unchanged genoview features
  carry (32)  pose + future trajectory + box(xy,ori)           -- box added to the query
  pick  (20)  pose + box(xy,ori)                                -- NO trajectory; box pos weighted
  place (20)  pose + box(xy,ori)                                -- NO trajectory
The box block is its PLANAR position (xy in the base frame) + orientation: box height is a
function of the phase and box velocity carries no signal the pose blocks lack, so neither
is matched on. pick and place share the 20-D layout but are separate
databases, so pick can weight the box
position more heavily (PICK_BOX_POS_WEIGHT) -- the entry is chosen mainly by where the box is.

The box rides the robot's gravity-aligned base frame while held (stored per frame as
boxLocal{Pos,Rot}); before pick contact and after place release it rests in the world.
"""
import numpy as np
from scipy.spatial import cKDTree

from . import config as C
from . import quat
from . import boxes
from .states import State
from .features import build_db, yaw_quat, FORWARD, HORIZONS, FPS
from .springs import (DecaySpringDamperPosition, DecaySpringDamperRotation,
                      TrajectorySpringPosition, TrajectorySpringRotation)

DT = C.DT
NDOF = 29
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def wrap_angle(a):
    """Wrap to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _yaw(q_wxyz):
    w, x, y, z = q_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


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
        # Per-phase normalized feature matrices + raw blocks for cross-database queries.
        self.Xloco, self.Xcarry = db["dbs"]["loco"]["X"], db["dbs"]["carry"]["X"]
        self.rawXpos, self.rawXvel = db["rawXpos"], db["rawXvel"]
        self.phase = lib["phase"]
        self.box_attach = lib["box_attach"]
        self.Ttimes = HORIZONS / FPS
        TAIL = C.SEARCH_TAIL

        # ---- Locomotion KD-trees: one per locomotion clip (phase==LOCO everywhere) ----
        self.loco_trees = []                                 # (rs, re, tree)
        for rs, re in zip(self.starts, self.stops):
            if self.phase[rs:re].any() or re - rs <= TAIL:   # skip box clips & tiny clips
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

        # Box resting orientation for the interactive spawn: the box pose in the base frame at
        # a representative pick entry == its world pose when the robot faces +x (yaw 0).
        self.box_spawn_rot = (self.boxLocalRot[self.pick_enter[0]].copy()
                              if len(self.pick_enter) else IDENTITY.copy())

        # The recorded stance-to-box relation at the pick entry (fold 0), inverted at
        # every B press to place the move-to-pick stance relative to the LIVE box.
        if len(self.pick_enter):
            e0 = int(self.pick_enter[0])
            bl = self.boxLocalPos[e0]                # box in the entry's sim-root frame
            self.stance_box_off = np.array([float(bl[0]), float(bl[1])])
            self.stance_box_yaw = _yaw(self.boxLocalRot[e0])
        else:
            self.stance_box_off = np.array([C.BOX_SPAWN_FWD, 0.0])
            self.stance_box_yaw = 0.0
        self.reset(start_frame)

    # --- state ---------------------------------------------------------------
    def reset(self, start_frame=None):
        if start_frame is None:
            start_frame = min(self.stops[0] - 1, self.starts[0] + 30)
        self.lo, self.hi = self._clip_bounds(start_frame)
        self.animFrame = int(start_frame)
        self.state = State.LOCOMOTION
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
        self.box_pending = False
        self.box_locked = 0
        # Move-to-pick (approach) state.
        self.move_timer = 0.0
        self.move_settle_t = 0.0
        self.on_rail = False
        self._move_box_xy = np.zeros(2)
        self.stance_xy = np.zeros(2)
        self.stance_yaw = 0.0
        self.route_wp = np.zeros(2)
        self.route_pts = []
        self.cmdVel = np.zeros(3)
        self.cmdFace = np.zeros(3)
        # Box world pose: a reachable distance in front of the robot's start facing, at its
        # resting height, oriented as it rests in the data (box_spawn_rot is in the base frame,
        # so composing with the start root gives the matching world orientation).
        self.boxPos = self.rootPos + quat.mul_vec(
            self.rootRot, np.array([C.BOX_SPAWN_FWD, C.BOX_SPAWN_LAT, 0.0]))
        self.boxPos[2] = C.BOX_REST_Z
        self.boxRot = quat.mul(self.rootRot, self.box_spawn_rot)
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

    # --- triggers ------------------------------------------------------------
    def trigger_box(self):
        """Request the box action (B): walk to the box and pick it up from locomotion, or
        place it while carrying. Pressing B during the walk cancels it. Honoured on the
        next step; a no-op while a pick/place is being ridden."""
        if self.state is State.MOVE_TO_PICK:
            self.state = State.LOCOMOTION
            return
        if self.box_locked == 0:
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
        self._maybe_trigger_box()

        # Move-to-pick drives itself: the player command is replaced by the walk
        # toward the stance planned from the live box pose. If the box moves
        # mid-approach (kicked, or fed from a live physical box), replan.
        if self.state is State.MOVE_TO_PICK:
            self.move_timer += DT
            if float(np.linalg.norm(self.boxPos[0:2] - self._move_box_xy)) > 0.10:
                self._start_move()
            desiredVel, desiredFace = self._steer_to_stance()
            if self._at_stance():
                self._enter_ride(self.pick_enter, self.pick_end_of,
                                 State.PICK)
                desiredVel = np.zeros(3)
                desiredFace = np.zeros(3)
            elif self.move_timer > C.MOVE_TIMEOUT:
                self.state = State.LOCOMOTION

        desiredVel = np.asarray(desiredVel, float)
        self.cmdVel = desiredVel.copy()
        self.cmdFace = np.asarray(desiredFace, float).copy()
        self._predict_trajectory(desiredVel, desiredFace)
        if self.state is State.MOVE_TO_PICK and np.linalg.norm(self.cmdVel) > 1e-6:
            self._path_taps()          # holding at the stance keeps the spring taps
        return self._query_from_trajectory(desiredVel)

    def _maybe_trigger_box(self):
        """Honour a pending B: locomotion -> walk to the pick stance; carry -> enter PLACE."""
        if not self.box_pending or self.box_locked > 0:
            self.box_pending = False
            return
        self.box_pending = False
        if self.state is State.LOCOMOTION and len(self.pick_enter):
            self._start_move()
        elif self.state is State.CARRY:
            self._enter_ride(self.place_enter, self.place_end_of, State.PLACE)

    # --- move-to-pick (approach heuristics, from motionmatching-g1-shelf) ----
    def _start_move(self):
        """Plan the approach: invert the recorded stance-to-box relation at the live box
        pose. The box's rotational symmetry gives SCENEBOT_ROT_FOLDS stance candidates
        around it; the nearest way-in point wins."""
        box_xy = self.boxPos[0:2]
        box_yaw = _yaw(self.boxRot)
        folds = max(1, C.SCENEBOT_ROT_FOLDS)
        best = None
        for k in range(folds):
            sy = wrap_angle(box_yaw + k * 2.0 * np.pi / folds - self.stance_box_yaw)
            rail = np.array([np.cos(sy), np.sin(sy)])
            perp = np.array([-rail[1], rail[0]])
            sxy = box_xy - self.stance_box_off[0] * rail - self.stance_box_off[1] * perp
            wp = sxy - C.MOVE_WAYIN * rail
            d = float(np.linalg.norm(wp - self.rootPos[0:2]))
            if best is None or d < best[0]:
                best = (d, sxy, sy, wp)
        _, self.stance_xy, self.stance_yaw, self.route_wp = best
        self._move_box_xy = self.boxPos[0:2].copy()
        self.state = State.MOVE_TO_PICK
        self.move_timer = 0.0
        self.move_settle_t = 0.0
        self.on_rail = False

    def _route_points(self):
        """The planned route as a polyline from the robot to the overshoot target:
        straight to the way-in point, a rounded corner there, then straight in along the
        rail. The corner arc keeps the heading turning continuously."""
        rail = np.array([np.cos(self.stance_yaw), np.sin(self.stance_yaw)])
        end = self.stance_xy + C.MOVE_OVERSHOOT * rail
        p0 = self.rootPos[0:2]
        if self.on_rail:
            return [p0, end]
        wp = self.route_wp
        d1 = wp - p0
        L1 = float(np.linalg.norm(d1))
        if L1 < 1e-6:
            return [p0, end]
        d1 = d1 / L1
        ang = float(np.arccos(np.clip(d1 @ rail, -1.0, 1.0)))
        if ang < 0.15:
            return [p0, end]
        # Round the corner at the way-in point with radius ~0.25 m; cap the fillet so
        # sharp approach angles keep a real straight leg.
        t = min(0.25 * np.tan(ang / 2.0), 0.3, 0.6 * L1,
                0.5 * float(np.linalg.norm(end - wp)))
        A = wp - d1 * t
        B = wp + rail * t
        corner = [(1 - s) ** 2 * A + 2 * (1 - s) * s * wp + s * s * B
                  for s in np.linspace(0.0, 1.0, 9)[1:-1]]
        return [p0, A] + corner + [B, end]

    def _steer_to_stance(self):
        """Walk the planned route toward a look-ahead point (facing the travel
        direction), then servo straight onto the stance for the last stretch and hold
        still inside the arrive radius so the root can settle before the pick."""
        rail = np.array([np.cos(self.stance_yaw), np.sin(self.stance_yaw)])
        rel = self.rootPos[0:2] - self.stance_xy
        along = float(rel @ rail)
        n = float(np.linalg.norm(rel - along * rail))
        stance_d = float(np.linalg.norm(rel))
        # Latch onto the final leg once the curve has merged with the rail; only fall
        # back off it on a big miss.
        if not self.on_rail:
            if along < -0.1 and n < 0.15:
                self.on_rail = True
                self.move_timer = 0.0      # fresh time budget for the last leg
        elif n > 0.45 or along > 0.25:
            self.on_rail = False

        face = np.array([rail[0], rail[1], 0.0])
        if stance_d < 0.8:
            # Endgame: servo straight at the stance (backward if overshot), facing
            # down the rail, and stop commanding well outside the stance so the
            # reference coasts to rest ON it. The floor 0.35 escapes the slow-walk
            # dead zone; the cap keeps arrival momentum low -- a tracking policy
            # lags decelerations, and a fast reference stop makes the physical
            # robot overshoot into the box (its feet kick it away).
            self.route_pts = [self.rootPos[0:2].copy(), self.stance_xy.copy()]
            vel = np.zeros(3)
            if stance_d > 0.18:
                vel[0:2] = -rel / stance_d * float(
                    np.clip(1.2 * stance_d, 0.35, 0.55))
            return vel, face

        route = self._route_points()
        self.route_pts = route
        # Route length and the look-ahead point ~0.45 m down the curve.
        look = route[-1]
        total = 0.0
        acc = 0.0
        prev = route[0]
        found = False
        for p in route[1:]:
            seg = float(np.linalg.norm(p - prev))
            total += seg
            if not found:
                acc += seg
                if acc >= 0.45:
                    look = p
                    found = True
            prev = p

        to = look - self.rootPos[0:2]
        dist = float(np.linalg.norm(to))
        vel = np.zeros(3)
        if dist > 1e-6:
            speed = float(np.clip(1.8 * total, 0.25, 1.2))
            vel[0:2] = to / dist * speed
            face = vel / (np.linalg.norm(vel) + 1e-9)
        return vel, face

    def _at_stance(self):
        """Arrived: standing at the stance, facing down the rail, root settled. A
        settled stop NEAR the stance also counts after a moment -- the entry match and
        the box-offset inertialization absorb a small residual, while waiting for a
        perfect stop can deadlock in the slow-walk dead zone."""
        rel = self.rootPos[0:2] - self.stance_xy
        dist = float(np.linalg.norm(rel))
        dyaw = abs(wrap_angle(self.stance_yaw - self.rootYaw))
        settled = float(np.linalg.norm(self.rootVel[0:2])) < C.MOVE_ARRIVE_SPEED
        if settled and dist < 0.30 and dyaw < C.MOVE_ARRIVE_YAW:
            self.move_settle_t += DT
        else:
            self.move_settle_t = 0.0
        if dist < C.MOVE_ARRIVE_NEAR and dyaw < C.MOVE_ARRIVE_YAW and settled:
            return True
        return self.move_settle_t > 1.0

    def _path_taps(self):
        """The future taps read straight off the planned route: walk the remaining path
        at the approach speed profile and sample the horizons."""
        pts = [p.copy() for p in self.route_pts[1:]]
        pos = self.rootPos[0:2].copy()
        heading = np.array([np.cos(self.stance_yaw), np.sin(self.stance_yaw)])
        k = 0
        for i in range(1, int(HORIZONS[-1]) + 1):
            rem, prev = 0.0, pos
            for p in pts:
                rem += float(np.linalg.norm(p - prev))
                prev = p
            adv = float(np.clip(1.8 * rem, 0.0, 1.2)) * DT
            while adv > 1e-9 and pts:
                seg = pts[0] - pos
                L = float(np.linalg.norm(seg))
                if L < 1e-9:
                    pts.pop(0)
                    continue
                heading = seg / L
                if adv < L:
                    pos = pos + heading * adv
                    adv = 0.0
                else:
                    pos = pts.pop(0)
                    adv -= L
            if k < len(HORIZONS) and i == int(HORIZONS[k]):
                self.Tpos[k] = np.array([pos[0], pos[1], 0.0])
                self.Tdir[k] = np.array([heading[0], heading[1], 0.0])
                k += 1

    def _enter_ride(self, enter_frames, end_of, state):
        """Nearest-neighbour match the live pose + box pose to the start of a pick/place phase
        (in that phase's own database), inertialize into it, and lock the playhead so the
        phase is ridden to its end."""
        if len(enter_frames) == 0:
            return
        dbname = state.name.lower()                  # State.PICK -> the 'pick' db
        Xq = self._query(dbname)
        Xdb = self.db["dbs"][dbname]["X"]
        entry = int(enter_frames[np.argmin(np.linalg.norm(Xdb[enter_frames] - Xq, axis=1))])
        end = int(end_of[entry])
        lo, _ = self._clip_bounds(entry)
        self._inertialize_into(entry, lo, end + 1)           # hi caps the ride at the phase end
        self.state = state
        self.box_locked = max(1, end - entry)
        self.searchTimer = C.SEARCH_TIME

    def _finish_ride(self):
        """Called when a PICK/PLACE ride ends: transition into the next searchable state."""
        if self.state is State.PICK:                       # pick -> carry
            res = self._best_carry()
            if res is not None:
                f, lo, hi = res
                self._inertialize_into(f, lo, hi)
                self.state = State.CARRY
        else:                                                # place -> locomotion
            f, lo, hi = self._best_loco()
            self._inertialize_into(f, lo, hi)
            self.state = State.LOCOMOTION
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
        """Live box pose in the controller base frame (yaw qh at the ground root).
        Position is planar (xy): the search ignores box height."""
        bp = quat.inv_mul_vec(qh, self.boxPos - self.rootPos)[0:2]
        br = quat.mul(quat.inv(qh), self.boxRot)
        return bp, quat.to_scaled_angle_axis(quat.abs(br))

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
        # ---- Search (skipped while riding a pick/place phase) ----
        if self.box_locked == 0 and self.searchTimer <= 0.0:
            if self.state is State.CARRY:
                self._search_carry()
            else:
                self._search_loco()
            self.searchTimer = C.SEARCH_TIME

        # ---- Advance the playhead within its current bounds ----
        self.animFrame = int(np.clip(self.animFrame + 1, self.lo, self.hi - 1))
        self.searchTimer -= DT
        if self.box_locked > 0:
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

        # Path snap: on the final approach leg the root is pinned to the rail, in
        # position and in heading -- the cross-track and yaw parts of the matched
        # motion are projected out.
        if self.state is State.MOVE_TO_PICK and self.on_rail:
            to = self.stance_xy - self.rootPos[0:2]
            if float(np.linalg.norm(to)) < C.SNAP_RADIUS:
                rail = np.array([np.cos(self.stance_yaw), np.sin(self.stance_yaw)])
                cross = to - float(to @ rail) * rail
                a = 1.0 - 0.5 ** (DT / C.SNAP_HALFLIFE)
                self.rootPos[0:2] += a * cross
                self.rootYaw += a * wrap_angle(self.stance_yaw - self.rootYaw)
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
        attached = (self.state in (State.PICK, State.CARRY, State.PLACE)
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

    def box_qpos(self):
        """Box freejoint qpos (7,) = world position + quaternion (wxyz), for the scene."""
        return np.concatenate([self.boxPos, self.boxRot])
