"""Motion matching + SONIC tracking for the box pick/carry/place task.

Shared machinery for the two variants (run_kinematic / run_grasp):
the mm_g1 matcher streamed as a 50 Hz reference (mm_stream.MMMotion), the SONIC
policy tracking it in MuJoCo, a commander that triggers the pick (the matcher's
MOVE_TO_PICK walks the approach itself), holds through carry, places, and walks
away, plus the five reference-correction modes (ref_modes). Subclasses set MODE and override _sync_box / _post_substep
for their box-handling strategy.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
os.environ.setdefault('MUJOCO_GL',
                      'glfw' if '--viewer' in sys.argv else 'egl')

import numpy as np
import mujoco

from sonic_tracking import params as P
from sonic_tracking.policy import SonicPolicy
from sonic_tracking.rotations import quat_rotate, quat_conjugate

from mm_g1 import config as C
from mm_g1.states import State
from mm_g1.data import load_library
from mm_g1.controller import MotionMatcher

import box_scene
import ref_modes as RM
from mm_stream import MMMotion, MM_FPS, POLICY_FPS

MARGIN = 50                       # 50 Hz frames kept ahead of the playhead
LOOKAHEAD_S = MARGIN / POLICY_FPS
REPLAN_MM_TICKS = 6               # matcher ticks per replan period (0.2 s)
GRIP_REF_DZ = 0.40                # reference lift before the grip is judged
GRIP_PHYS_DZ = 0.04               # physical lift that counts as gripped
FALL_Z = 0.28                     # pelvis below this counts as fallen (squats go low)
FALL_TILT = -0.10                 # base-frame gravity z above this = tipped right over
FALL_REF_TILT = 0.50              # tilt allowed BEYOND the reference pose's own tilt
GHOST_RGBA = np.array([1.0, 0.75, 0.2, 0.55], np.float32)
CONTACT_RGBA = np.array([0.25, 0.95, 0.35, 0.55], np.float32)  # ref box while hand contact is labeled
_EYE3 = np.eye(3).ravel()


class Demo:
    MODE = None                   # 'kinematic' | 'grasp'

    WALK_AWAY_DIST = 1.3

    def __init__(self, args):
        self.args = args
        scene_xml = P.G1_SCENE_XML if args.robot == 'sonic' else os.path.join(
            ROOT, 'assets', 'scenebot', 'scene_robot_only.xml')
        self.model, self.ids = box_scene.build_model(
            scene_xml, self.MODE, box_mass=args.box_mass,
            off_w=args.width, off_h=args.height,
            box_scale=args.box_scale, box_type=args.box,
            box_friction=args.box_friction)
        self.model.opt.timestep = P.CONTROL_DT / args.substeps
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.q_at = np.array([m.joint(n).qposadr[0]
                              for n in P.JOINT_NAMES_MUJOCO])
        self.dq_at = np.array([m.joint(n).dofadr[0]
                               for n in P.JOINT_NAMES_MUJOCO])
        self.bq = self.ids['box_qpos_at']
        self.bd = self.ids['box_dof_at']
        self.kps, self.kds = P.KPS.copy(), P.KDS.copy()

        lib = load_library()
        self.matcher = MotionMatcher(lib)

        self.t = 0.0
        self.lift_seen = False
        self.last_up_time = None
        self.carry_start = None
        self.carry_peak = 0.0
        self.place_done = False
        self.mm_placed = False
        self._mm_prev = None
        self._prev_mm_held = False
        self._hold_start_f = np.inf
        self.max_box_z = 0.0
        self.fallen, self.fall_time = False, None

        self.motion = MMMotion(self.matcher, self._command)

        q0 = self.motion.qpos[0]
        d = self.data
        d.qpos[:] = 0.0
        d.qpos[0:7] = q0[0:7]
        d.qpos[2] = 0.80
        d.qpos[self.q_at] = q0[7:36]
        d.qpos[self.bq:self.bq + 7] = q0[36:43]
        mujoco.mj_forward(self.model, d)

        self.policy = SonicPolicy(variant=args.sonic, device='cpu')
        self.policy.streaming = True
        self.policy.set_motion(self.motion)
        # One shared world frame: pin heading alignment to identity so the
        # anchor-orientation observation is true heading error.
        self.policy.heading_init_base_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.policy.init_ref_root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.policy.delta_heading = 0.0
        self.policy.reinitialize_heading = False
        self.policy._update_heading = lambda base_quat: None

        self.ref_mode = args.ref_mode
        self.seeder = self.anchor = None
        if self.ref_mode in RM.REPLAN_MODES:
            self.seeder = RM.RootSeeder(self.ref_mode, self.matcher, self.data,
                                        self.motion, gain=args.replan_gain,
                                        q_at=self.q_at, dq_at=self.dq_at)
        else:
            self.anchor = RM.ContinuousAnchor(self.matcher, self.data,
                                              self.motion, self.policy,
                                              gain=args.anchor_gain)
            self.motion.pre_tick = self.anchor.before_matcher_tick
        self.started = False
        self.next_replan = 0.0
        self.gdata = mujoco.MjData(self.model)       # ghost FK only
        self.ghost_bodies = [
            b for b in range(1, self.model.nbody)
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            != 'largebox']

        self._open_outputs()
        self.viewer = None
        self.show_ui = not args.no_ui
        if args.viewer:
            import mujoco.viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(
                self.model, self.data,
                show_left_ui=False, show_right_ui=False)
            self.viewer.cam.distance, self.viewer.cam.azimuth = 2.8, 120.0
            self.viewer.cam.elevation = -18.0
            self._vlook = self._focus_point()    # start ON target: no initial glide
            self.viewer.cam.lookat[:] = self._vlook

        self.setup_extra()
        # No plain-PD settle (it tips over): run the policy paused at frame 0.
        for _ in range(int(1.0 / P.CONTROL_DT)):
            self.step_physics()
            if self.viewer is not None:
                self._follow_cam()
                self._update_overlay()
                self.viewer.sync()
        self.policy.start_play()
        self.started = True

    # SONIC never observes the reference's xy position, so the commander steers
    # by the PHYSICAL robot; the --ref-mode machinery keeps the reference root
    # consistent with the robot. Decisions use only physical state or committed
    # stream frames, so speculative replan rollouts stay side-effect free.
    def _command(self, mm):
        d = self.data
        robot_xy = d.qpos[0:2].copy()
        box_xy = d.qpos[self.bq:self.bq + 2].copy()

        if self.MODE != 'kinematic':
            if not mm.box_held:
                # Live box feedback: the physical box is the truth while not held.
                mm.boxPos[:] = d.qpos[self.bq:self.bq + 3]
                mm.boxRot[:] = d.qpos[self.bq + 3:self.bq + 7]
            else:
                if not self._prev_mm_held:
                    self._hold_start_f = self.motion.timesteps
                self._check_grip(mm)
            self._prev_mm_held = mm.box_held

        st = mm.state
        # Matcher-side placement latch: the matcher runs ~1 s ahead of the
        # tracked frame, so a fresh LOCOMOTION here must not re-trigger a pick
        # while the physical robot is still finishing the place.
        if self._mm_prev is State.PLACE and st is State.LOCOMOTION:
            if (self.last_up_time is not None
                    and self.t - self.last_up_time < 2.0):
                self.mm_placed = True
        self._mm_prev = st
        if st in (State.PICK, State.PLACE):
            return np.zeros(3), np.zeros(3)
        if st is State.CARRY:
            if (self.carry_start is not None
                    and self.t - self.carry_start > self.args.carry_seconds):
                mm.trigger_box()               # set it down
            return np.zeros(3), np.zeros(3)

        if self.place_done:
            # Gentle walk-away: a brisk turn right next to a light box kicks it.
            away = robot_xy - box_xy
            n = float(np.linalg.norm(away))
            u = away / max(n, 1e-6)
            face = np.array([u[0], u[1], 0.0])
            if n > self.WALK_AWAY_DIST:
                return np.zeros(3), face
            speed = 0.4 if n < 1.0 else 0.7
            return speed * face, face

        # The walk to the box is the matcher's own MOVE_TO_PICK: B plans the
        # approach from the live box pose and self-drives it (the commander's
        # velocity is ignored while it runs). Pressing B again would CANCEL
        # it, so only trigger from LOCOMOTION -- once to start, and again
        # whenever a dropped grip has knocked the matcher back to locomotion.
        if st is State.MOVE_TO_PICK:
            return np.zeros(3), np.zeros(3)
        if not self.mm_placed:
            mm.trigger_box()
        return np.zeros(3), np.zeros(3)

    def _check_grip(self, mm):
        """The matcher's box is kinematic: once its clip marks the box held it
        rides up regardless of physics. Judged at the TRACKED frame (the
        matcher runs ~1 s ahead): if the reference box has lifted but the
        physical box stayed on the ground, the grip failed -- drop the
        matcher's belief and fall back to locomotion so the retry is
        immediate. The next replan truncates the stale carry horizon. Only
        frames of the CURRENT hold episode are judged -- older tracked frames
        still show the previous attempt and must not abort this one."""
        f = min(int(self.policy.current_frame), self.motion.timesteps - 1)
        if f < self._hold_start_f:
            return
        ref_held = self.motion.meta_at(f)[3]
        ref_z = float(self.motion.qpos[f][38])
        phys_z = float(self.data.qpos[self.bq + 2])
        if (ref_held and ref_z > C.BOX_REST_Z + GRIP_REF_DZ
                and phys_z < C.BOX_REST_Z + GRIP_PHYS_DZ):
            mm.box_locked = 0
            mm.box_pending = False
            mm.state = State.LOCOMOTION
            mm.box_held = False
            mm.boxPos[:] = self.data.qpos[self.bq:self.bq + 3]
            mm.boxRot[:] = self.data.qpos[self.bq + 3:self.bq + 7]
            mm.searchTimer = 0.0
            if self.t - getattr(self, '_last_grip_abort', -1.0) > 0.5:
                self._last_grip_abort = self.t
                print(f'[{self.t:6.2f}s] GRIP FAILED (ref box z {ref_z:.2f}, '
                      f'physical {phys_z:.2f}) -> retry')

    def _replan(self):
        """Truncate the stale horizon, re-seed the matcher from the robot,
        roll it forward; only the first replan period is committed."""
        f = int(self.policy.current_frame)
        self.motion.truncate(f + 1)
        self.seeder.seed(f)
        self.matcher.searchTimer = 0.0
        commit = f + 1 + int(REPLAN_MM_TICKS / MM_FPS * POLICY_FPS)
        horizon = f + 1 + int((LOOKAHEAD_S + REPLAN_MM_TICKS / MM_FPS)
                              * POLICY_FPS)
        keep = None
        while self.motion.timesteps < horizon:
            self.motion.ensure(self.motion.timesteps + 1)
            if keep is None and self.motion.timesteps >= commit:
                keep = (RM.snapshot(self.matcher), self.motion.time_mark())
        if keep is not None:
            RM.restore(self.matcher, keep[0])
            self.motion.time_restore(keep[1])

    # Box-handling hooks, overridden per variant.
    def setup_extra(self):
        """Before the warmup loop."""

    def _adjust_target(self, target, f):
        """PD target hook (MuJoCo joint order); `f` is the tracked frame."""
        return target

    def _sync_box(self, f):
        """Before the physics substeps; `f` is the tracked reference frame."""

    def _post_substep(self, f):
        """After each mj_step."""

    def step_physics(self):
        d = self.data
        if self.seeder is not None and self.started:
            robot_time = self.policy.current_frame / POLICY_FPS
            if robot_time >= self.next_replan:
                self._replan()
                self.next_replan = robot_time + REPLAN_MM_TICKS / MM_FPS
        self.motion.ensure(self.policy.current_frame + MARGIN)
        if self.anchor is not None:
            self.anchor.record_applied()
        target = self.policy.step(d.qpos[3:7].copy(), d.qvel[3:6].copy(),
                                  d.qpos[self.q_at].copy(),
                                  d.qvel[self.dq_at].copy())
        f = min(int(self.policy.current_frame), self.motion.timesteps - 1)
        target = self._adjust_target(target, f)
        self._sync_box(f)
        for _ in range(self.args.substeps):
            d.ctrl[:29] = (self.kps * (target - d.qpos[self.q_at])
                           - self.kds * d.qvel[self.dq_at])
            mujoco.mj_step(self.model, d)
            self._post_substep(f)

        _, _, state, held, _ = self.motion.meta_at(f)
        box_z = float(d.qpos[self.bq + 2])
        self.max_box_z = max(self.max_box_z, box_z)
        if box_z > C.BOX_REST_Z + 0.25:
            self.lift_seen = True
            self.last_up_time = self.t
        if state is State.CARRY and held and self.carry_start is None:
            self.carry_start = self.t
            self.carry_peak = box_z
            print(f'[{self.t:6.2f}s] CARRY (box z {box_z:.2f} m)')
        if self.carry_start is not None:
            self.carry_peak = max(self.carry_peak, box_z)
        if state is State.LOCOMOTION and not held and not self.place_done:
            if (self.carry_start is not None
                    and self.carry_peak > C.BOX_REST_Z + 0.25):
                self.place_done = True
                print(f'[{self.t:6.2f}s] PLACED (box z {box_z:.2f} m)')
            else:
                self.carry_start = None          # a failed pick may retry

    def _open_outputs(self):
        a = self.args
        self.renderer = self.writer = self.cam = None
        self.look = self._focus_point()
        if a.no_video:
            return
        import imageio
        os.makedirs(os.path.dirname(a.video_path), exist_ok=True)
        self.renderer = mujoco.Renderer(self.model, a.height, a.width)
        self.writer = imageio.get_writer(a.video_path, fps=int(POLICY_FPS),
                                         codec='libx264', quality=8,
                                         macro_block_size=None)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.azimuth, self.cam.elevation = \
            3.6, -35.0, -16.0

    def _draw_ghost(self, scn):
        f = min(int(self.policy.current_frame), self.motion.timesteps - 1)
        gq = self.motion.qpos[f]
        g = self.gdata
        g.qpos[:] = 0.0
        g.qpos[3] = 1.0
        g.qpos[0:7] = gq[0:7]
        g.qpos[self.q_at] = gq[7:36]
        mujoco.mj_kinematics(self.model, g)
        for b in self.ghost_bodies:
            pa = self.model.body_parentid[b]
            if pa == 0 or scn.ngeom >= scn.maxgeom:
                continue
            a, c = g.xpos[pa], g.xpos[b]
            if np.linalg.norm(c - a) < 1e-6:
                continue
            gm = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(gm, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                np.zeros(3), np.zeros(3), _EYE3, GHOST_RGBA)
            mujoco.mjv_connector(gm, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.025,
                                 np.asarray(a, float), np.asarray(c, float))
            scn.ngeom += 1
        if self.MODE != 'kinematic' and scn.ngeom < scn.maxgeom:
            # the reference box, so grasp tracking error is visible;
            # the carton is tilted in its local frame, so compose its OBB
            gm = scn.geoms[scn.ngeom]
            gc, gmat, ghalf = self.ids['ghost']
            lc, rc = self.motion.meta_at(f)[4]
            mat = np.empty(9)
            mujoco.mju_quat2Mat(mat, np.asarray(gq[39:43], float))
            R = mat.reshape(3, 3) @ gmat
            pos = gq[36:39] + mat.reshape(3, 3) @ gc
            mujoco.mjv_initGeom(gm, mujoco.mjtGeom.mjGEOM_BOX, ghalf,
                                np.asarray(pos, float), R.ravel(),
                                CONTACT_RGBA if (lc or rc) else GHOST_RGBA)
            scn.ngeom += 1

    def _overlay_text(self):
        f = min(int(self.policy.current_frame), self.motion.timesteps - 1)
        cid, fic, state, held, (lc, rc) = self.motion.meta_at(f)
        lib = self.matcher.lib
        clip, length = lib['clip_names'][cid], int(lib['lengths'][cid])
        speed = float(np.linalg.norm(self.data.qvel[0:2]))
        head = state.name if state is not State.LOCOMOTION else \
            ('WALK' if speed > 0.1 else 'IDLE')
        box_z = float(self.data.qpos[self.bq + 2])
        title = f'{head}   {speed:.1f} m/s   [{self.MODE}]'
        body = (f'clip [{cid}]: {clip}\n'
                f'frame: {fic}/{length - 1}  (tracked ref frame {f})\n'
                f'box: z {box_z:.2f} m'
                f'  ref-{"held" if held else "resting"}'
                f'  contact [{"L" if lc else "-"}{"R" if rc else "-"}]\n'
                f'ref-mode: {self.ref_mode}')
        return title, body

    def _update_overlay(self):
        if not self.show_ui or self.viewer is None:
            return
        self.viewer.set_texts((None, None, *self._overlay_text()))

    def _focus_point(self):
        """Camera aim: between the robot and the box, at chest height."""
        box_xy = self.data.qpos[self.bq:self.bq + 2]
        return np.array([0.55 * self.data.qpos[0] + 0.45 * box_xy[0],
                         0.55 * self.data.qpos[1] + 0.45 * box_xy[1], 0.7])

    def _follow_cam(self):
        """Track the action with the live viewer camera (lookat only, so
        mouse orbit / zoom still work)."""
        self._vlook += 0.06 * (self._focus_point() - self._vlook)
        self.viewer.cam.lookat[:] = self._vlook

    def render(self):
        if self.renderer is None:
            return
        self.look += 0.06 * (self._focus_point() - self.look)
        self.cam.distance += 0.03 * (3.6 - self.cam.distance)
        self.cam.lookat[:] = self.look
        self.renderer.update_scene(self.data, camera=self.cam)
        self._draw_ghost(self.renderer.scene)
        if self.show_ui:
            # mujoco.Renderer has no overlay API: replicate its RGB render()
            # with mjr_overlay injected between the render and the readback.
            r = self.renderer
            if r._gl_context:
                r._gl_context.make_current()
            mujoco.mjr_render(r._rect, r._scene, r._mjr_context)
            title, body = self._overlay_text()
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                               mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               r._rect, title, body, r._mjr_context)
            frame = np.empty((r._height, r._width, 3), np.uint8)
            mujoco.mjr_readPixels(frame, None, r._rect, r._mjr_context)
            frame = np.flipud(frame)
        else:
            frame = self.renderer.render()
        self.writer.append_data(frame)

    def run(self):
        a = self.args
        wall = time.time()
        for tick in range(int(a.max_seconds / P.CONTROL_DT)):
            self.t = tick * P.CONTROL_DT
            self.step_physics()

            # The pick/place squat pitches the pelvis far forward, so tilt is
            # judged against the REFERENCE pose, not an absolute threshold.
            down = np.array([0.0, 0.0, -1.0])
            gravity = quat_rotate(quat_conjugate(self.data.qpos[3:7]), down)
            ref = self.motion.qpos[min(int(self.policy.current_frame),
                                       self.motion.timesteps - 1)]
            gref = quat_rotate(quat_conjugate(ref[3:7]), down)
            if not self.fallen and (self.data.qpos[2] < FALL_Z
                                    or gravity[2] > FALL_TILT
                                    or gravity[2] - gref[2] > FALL_REF_TILT):
                self.fallen, self.fall_time = True, self.t
                print(f'[{self.t:6.2f}s] ROBOT FELL')

            self.render()
            if self.viewer is not None:
                if not self.viewer.is_running():
                    break
                scn = getattr(self.viewer, 'user_scn', None)
                if scn is not None:
                    scn.ngeom = 0
                    self._draw_ghost(scn)
                self._follow_cam()
                self._update_overlay()
                self.viewer.sync()
            if self.fallen and (self.t - self.fall_time) > 2.0:
                break
            if self.place_done and self._robot_box_dist() > self.WALK_AWAY_DIST:
                break
            if tick % 100 == 0:
                print(f'  t={self.t:5.1f}s mm={self.matcher.state.name:12s} '
                      f'x={self.data.qpos[0]:5.2f} '
                      f'box_z={self.data.qpos[self.bq + 2]:5.2f} '
                      f'held={self.motion.meta_at(min(int(self.policy.current_frame), self.motion.timesteps - 1))[3]}',
                      flush=True)
        return self.finish(wall)

    def _robot_box_dist(self):
        return float(np.linalg.norm(self.data.qpos[0:2]
                                    - self.data.qpos[self.bq:self.bq + 2]))

    def finish(self, wall):
        if self.writer is not None:
            self.writer.close()
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
        for line in (self.seeder or self.anchor).summary_lines():
            print(line)
        d = self.data
        box_z = float(d.qpos[self.bq + 2])
        up = np.empty(9)
        mujoco.mju_quat2Mat(up, np.asarray(d.qpos[self.bq + 3:self.bq + 7],
                                           float))
        upright = abs(up[8]) > 0.85          # resting on a z face (either one)
        placed = abs(box_z - C.BOX_REST_Z) < 0.08 and upright
        away = self._robot_box_dist()
        ok = (self.lift_seen and placed and self.place_done
              and not self.fallen and away > 1.0)
        print(f'[demo:{self.MODE}] {"SUCCESS" if ok else "INCOMPLETE"} -- '
              f'{self.t:.1f} s sim, {time.time() - wall:.0f} s wall, '
              f'lifted={self.lift_seen} (max z {self.max_box_z:.2f} m), '
              f'placed={placed} (z {box_z:.2f}, upright={upright}), '
              f'fallen={self.fallen}, robot-box dist {away:.2f} m')
        if self.writer is not None:
            print(f'[demo] wrote {self.args.video_path}')
        return 0 if ok else 1


def build_argparser(video_name):
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--viewer', action='store_true',
                    help='watch live (needs a display)')
    ap.add_argument('--no-ui', action='store_true')
    ap.add_argument('--max-seconds', type=float, default=40.0)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--ref-mode', choices=RM.MODES, default='anchor')
    ap.add_argument('--sonic', choices=list(P.SONIC_VARIANTS),
                    default=P.DEFAULT_VARIANT,
                    help='SONIC checkpoint: release (v1.0), low_latency '
                         '(~80 ms lookahead), sonic_v1_1 (heading-normalized '
                         'target orientations)')
    ap.add_argument('--robot', choices=['scenebot', 'sonic'], default='scenebot',
                    help="G1 model: SceneBot's flat-hand G1 (the model the "
                         'motion + contact labels were made with; its own '
                         "palm collision) or NVIDIA's 29-DoF SONIC scene "
                         '(visual-only hands, capsule pads bolted on)')
    ap.add_argument('--anchor-gain', type=float, default=0.20)
    ap.add_argument('--replan-gain', type=float, default=0.738)
    ap.add_argument('--box', choices=['scenebot', 'carton'],
                    default='scenebot',
                    help="physical box: the SceneBot free box the library is "
                         'baked for (C.BOX_HALF), or the OmniRetarget MEDICINE '
                         'carton mesh (a different, larger size)')
    ap.add_argument('--box-mass', type=float, default=0.5)
    ap.add_argument('--substeps', type=int, default=20,
                    help='physics substeps per 50 Hz control tick (timestep = '
                         '0.02/substeps); the stiff light-box contact needs '
                         '>= 16 to integrate stably')
    ap.add_argument('--box-friction', type=float, default=1.5,
                    help='sliding friction of the box geom; contacts use the '
                         'pair MAXIMUM, so this alone sets box-hand friction')
    ap.add_argument('--box-scale', type=float, default=1.0,
                    help='scale the physical box only (reference motion '
                         'unchanged)')
    ap.add_argument('--carry-seconds', type=float, default=3.0,
                    help='how long to hold the box before setting it down')
    ap.add_argument('--video', default=os.path.join(HERE, 'out', video_name))
    return ap


def run_main(demo_cls, video_name, extra_args=None):
    ap = build_argparser(video_name)
    if extra_args:
        extra_args(ap)
    args = ap.parse_args()
    args.video_path = args.video
    return demo_cls(args).run()
