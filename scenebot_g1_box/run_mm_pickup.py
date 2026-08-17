"""This repo's motion-matched box pick/carry/place reference, tracked by the
SceneBot policy instead of SONIC.

The mm_g1 matcher streams the 50 Hz reference (mm_stream.MMMotion: robot
qpos + box pose), and each frame is converted to a SceneBot stream packet:
leg joint targets, VR 3-point wrist/head targets via FK on the SceneBot G1,
motion anchor = the reference root, and a synthesized contact label (feet
during pick/place, wrists while reaching or holding). The commander steers
the matcher by the REFERENCE root (open loop w.r.t. physics); the policy
observes the anchor error and follows.

Box variants:
  kinematic (default) -- box has no collision, teleported to the reference
      pose each substep; tests pure motion tracking.
  grasp -- free carton, hand friction only, like sonic_g1_box/run_grasp.

The pickup reference here (OmniRetarget carton clips) is OUT of the SceneBot
policy's training distribution; success is an empirical question this
script answers with exit code 0/1 and tracking stats.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
sys.path.insert(0, os.path.join(ROOT, 'sonic_g1_box'))
os.environ.setdefault('MUJOCO_GL',
                      'glfw' if '--viewer' in sys.argv else 'egl')

import numpy as np
import mujoco

from scenebot_tracking import params as P
from scenebot_tracking.motion_graph import LOWER_IDX, VR_OFFSETS
from scenebot_tracking.policy import ScenebotPolicy
from scenebot_tracking.rotations import (wxyz_to_xyzw, xyzw_to_wxyz,
                                         quat_conj_xyzw, quat_mul_xyzw,
                                         quat_rotate_xyzw)

from mm_g1 import config as C
from mm_g1 import quat as MQ
from mm_g1.data import load_library
from mm_g1.controller import MotionMatcher

import box_scene
from mm_stream import MMMotion, MM_FPS, POLICY_FPS

MARGIN = 20
GHOST_RGBA = np.array([1.0, 0.75, 0.2, 0.55], np.float32)
_EYE3 = np.eye(3).ravel()
VR_BODY_NAMES = ['left_wrist_yaw_link', 'right_wrist_yaw_link', 'torso_link']


class MMPacketAdapter:
    """One SceneBot stream packet per 50 Hz reference frame."""

    def __init__(self, motion, model, q_at, rq):
        self.motion = motion
        self.model = model
        self.q_at = q_at
        self.rq = rq
        self.fk = mujoco.MjData(model)
        self.body_ids = [model.body(n).id for n in VR_BODY_NAMES]
        self.pelvis_id = model.body('pelvis').id

    def packet(self, f):
        mo = self.motion
        f = min(f, mo.timesteps - 1)
        q = mo.qpos[f]
        jp = mo.joint_pos[f]                 # (29,) isaac order
        jv = mo.joint_vel[f]
        lower_cmd = np.concatenate([jp[LOWER_IDX], jv[LOWER_IDX]]).astype(np.float32)

        d = self.fk
        d.qpos[:] = 0.0
        d.qpos[self.rq:self.rq + 7] = q[0:7]
        d.qpos[self.q_at] = q[7:36]
        mujoco.mj_kinematics(self.model, d)
        root_p = d.xpos[self.pelvis_id].copy()
        root_inv = quat_conj_xyzw(wxyz_to_xyzw(d.xquat[self.pelvis_id]))
        vr_pos = np.zeros(9, np.float32)
        vr_orn = np.zeros(12, np.float32)
        for i, b in enumerate(self.body_ids):
            body_q = wxyz_to_xyzw(d.xquat[b])
            off_w = quat_rotate_xyzw(body_q, VR_OFFSETS[i])
            rel = d.xpos[b] + off_w - root_p
            vr_pos[i * 3:i * 3 + 3] = quat_rotate_xyzw(root_inv, rel)
            vr_orn[i * 4:i * 4 + 4] = xyzw_to_wxyz(
                quat_mul_xyzw(root_inv, body_q))

        _, _, state, held = mo.meta_at(f)
        skill = state in ('PICK', 'PLACE')
        wrist = 1.0 if (held or skill) else 0.0
        foot = 1.0 if skill else 0.0
        mask = np.array([foot, foot, wrist, wrist, 0.0], np.float32)

        return {
            'lower_cmd': lower_cmd,
            'vr_3point_pos_l': vr_pos,
            'vr_3point_orn_l': vr_orn,
            'contact_mask': mask,
            'motion_anchor_pos_w': q[0:3].astype(np.float32),
            'motion_anchor_orn_w': wxyz_to_xyzw(q[3:7]).astype(np.float32),
        }


class Demo:
    STAND_DIST = 0.7
    WALK_AWAY_DIST = 1.3

    def __init__(self, args):
        self.args = args
        self.mode = args.mode
        self.model, self.ids = box_scene.build_model(
            os.path.join(P.ASSET_DIR, 'scene_robot_only.xml'), self.mode,
            box_mass=args.box_mass, off_w=args.width, off_h=args.height)
        self.model.opt.timestep = P.SIM_DT
        self.data = mujoco.MjData(self.model)
        m = self.model

        free_types = {mujoco.mjtJoint.mjJNT_FREE}
        hinge = [j for j in range(m.njnt) if m.jnt_type[j] not in free_types]
        assert len(hinge) == 29, len(hinge)
        self.q_at = np.array([m.jnt_qposadr[j] for j in hinge])
        self.dq_at = np.array([m.jnt_dofadr[j] for j in hinge])
        fb = m.joint('floating_base_joint')
        self.rq = int(fb.qposadr[0])
        self.rd = int(fb.dofadr[0])
        self.bq = self.ids['box_qpos_at']
        self.bd = self.ids['box_dof_at']

        # like sonic_g1_box/run_grasp: tracking alone produces no squeeze
        # pressure, so grasp mode stiffens the arms and biases the shoulder
        # rolls inward while the reference holds the box
        self.kps, self.kds = P.KPS.copy(), P.KDS.copy()
        if self.mode == 'grasp':
            self.kps[15:29] *= args.arm_gain
            self.kds[15:29] *= np.sqrt(args.arm_gain)

        lib = load_library()
        self.matcher = MotionMatcher(lib)

        self.t = 0.0
        self.frame_f = 0.0
        self.frame = 0
        self.carry_start = None
        self.place_done = False
        self.pick_triggered = False
        self._mm_prev = None
        self._place_time = 0.0
        self.attached = False
        self.min_root_z = 10.0
        self.max_xy_err = 0.0
        self.max_box_z = 0.0
        self.fallen, self.fall_time = False, None

        self.motion = MMMotion(self.matcher, self._command)
        self.adapter = MMPacketAdapter(self.motion, self.model, self.q_at,
                                       self.rq)
        self.policy = ScenebotPolicy()

        q0 = self.motion.qpos[0]
        d = self.data
        d.qpos[:] = 0.0
        d.qpos[self.rq:self.rq + 7] = q0[0:7]
        d.qpos[self.rq + 2] = 0.80
        d.qpos[self.q_at] = q0[7:36]
        d.qpos[self.bq:self.bq + 7] = q0[36:43]
        mujoco.mj_forward(self.model, d)
        self.start_xy = q0[0:2].copy()

        self.gdata = mujoco.MjData(self.model)
        self.ghost_bodies = [
            b for b in range(1, self.model.nbody)
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            != 'largebox']

        self._open_outputs()
        self.viewer = None
        if args.viewer:
            import mujoco.viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance, self.viewer.cam.azimuth = 3.6, -35.0
            self.viewer.cam.elevation = -16.0
            self.viewer.cam.lookat[:] = [0.8, 0.0, 0.7]

    # Steer the matcher by its OWN root (deterministic, physics-free); the
    # policy follows via the anchor-position observation.
    def _command(self, mm):
        ref = self.motion._mm_t1 if self.motion is not None else None
        if ref is None or self.motion._ticks / MM_FPS < self.args.settle_seconds:
            return np.zeros(3), np.zeros(3)
        ref_xy = ref[0:2]
        ref_speed = float(np.linalg.norm(ref_xy - self.motion._mm_t0[0:2])
                          * MM_FPS)
        box_xy = mm.boxPos[:2].copy()

        st = mm.state_name()
        if self._mm_prev == 'PLACE' and st == 'LOCOMOTION':
            self.place_done = True
            self._place_time = self.motion._ticks / MM_FPS
        self._mm_prev = st
        if st in ('PICK', 'PLACE'):
            return np.zeros(3), np.zeros(3)
        if st == 'CARRY':
            if (self.carry_start is not None
                    and self.motion._ticks / MM_FPS - self.carry_start
                    > self.args.carry_seconds):
                mm.trigger_box()
            elif self.carry_start is None:
                self.carry_start = self.motion._ticks / MM_FPS
            return np.zeros(3), np.zeros(3)

        if self.place_done:
            away = ref_xy - box_xy
            n = float(np.linalg.norm(away))
            u = away / max(n, 1e-6)
            face = np.array([u[0], u[1], 0.0])
            # stand up and settle before walking off; the place clip ends in
            # a deep squat and the policy needs a beat to recover
            if self.motion._ticks / MM_FPS - self._place_time \
                    < self.args.post_place_pause:
                return np.zeros(3), face
            if n > self.WALK_AWAY_DIST:
                return np.zeros(3), face
            return 0.35 * face, face

        to_box = box_xy - ref_xy
        dist = float(np.linalg.norm(to_box))
        u_appr = box_xy - self.start_xy
        u_appr /= max(float(np.linalg.norm(u_appr)), 1e-6)
        spot = box_xy - self.STAND_DIST * u_appr
        to_spot = spot - ref_xy
        far = float(np.linalg.norm(to_spot))
        face = np.array([to_box[0] / max(dist, 1e-6),
                         to_box[1] / max(dist, 1e-6), 0.0])
        if far > 0.06 and dist > self.STAND_DIST + 0.05:
            speed = float(np.clip(1.4 * far, 0.35, 0.7))
            return np.array([*(to_spot / far * speed), 0.0]), face
        if ref_speed < 0.15:
            mm.trigger_box()
            self.pick_triggered = True
        return np.zeros(3), face

    def _read_state(self):
        d = self.data
        quat_wxyz = d.qpos[self.rq + 3:self.rq + 7]
        return {
            'q': d.qpos[self.q_at].copy(),
            'dq': d.qvel[self.dq_at].copy(),
            'root_pos': d.qpos[self.rq:self.rq + 3].copy(),
            'root_orn_xyzw': np.array([quat_wxyz[1], quat_wxyz[2],
                                       quat_wxyz[3], quat_wxyz[0]]),
            'root_vel': d.qvel[self.rd:self.rd + 3].copy(),
            'omega': d.qvel[self.rd + 3:self.rd + 6].copy(),
        }

    def step_control(self):
        self.motion.ensure(self.frame + MARGIN)
        pkt = self.adapter.packet(self.frame)
        self.policy.ingest(pkt)
        target = self.policy.step(self._read_state())
        d = self.data
        f = min(self.frame, self.motion.timesteps - 1)
        if self.mode == 'grasp' and self.motion.meta_at(f)[3]:
            target = target.copy()
            target[16] -= self.args.squeeze     # left shoulder roll inward
            target[23] += self.args.squeeze     # right shoulder roll inward
        if self.mode == 'weld':
            self._sync_weld(f)
        for _ in range(P.DECIMATION):
            tau = (self.kps * (target - d.qpos[self.q_at])
                   - self.kds * d.qvel[self.dq_at])
            d.ctrl[:29] = np.clip(tau, -P.TORQUE_LIMIT, P.TORQUE_LIMIT)
            mujoco.mj_step(self.model, d)
            if self.mode == 'kinematic':
                d.qpos[self.bq:self.bq + 7] = self.motion.qpos[f][36:43]
                d.qvel[self.bd:self.bd + 6] = 0.0

        # Track the pick/place squat at half rate, like the SceneBot demo
        # plays its own pickup clip (pickupForwardStepScale 0.5); the ref
        # squat/stand-up is otherwise too fast for the policy. Only while
        # the reference is near-stationary -- slowing while it translates
        # makes the full-speed leg commands fight the half-speed anchor.
        _, _, state, _ = self.motion.meta_at(f)
        ref_z = float(self.motion.qpos[f][2])
        nxt = self.motion.qpos[min(f + 1, self.motion.timesteps - 1)]
        ref_speed = float(np.linalg.norm(nxt[0:2] - self.motion.qpos[f][0:2])
                          * POLICY_FPS)
        slow = ((state in ('PICK', 'PLACE') or ref_z < 0.68)
                and ref_speed < 0.25)
        self.frame_f += 0.5 if slow else 1.0
        self.frame = int(self.frame_f)

        ref = self.motion.qpos[f]
        xy_err = float(np.linalg.norm(d.qpos[self.rq:self.rq + 2] - ref[0:2]))
        self.max_xy_err = max(self.max_xy_err, xy_err)
        self.min_root_z = min(self.min_root_z, float(d.qpos[self.rq + 2]))
        self.max_box_z = max(self.max_box_z, float(d.qpos[self.bq + 2]))

    def _palm_box_dist(self):
        box_c = self.data.qpos[self.bq:self.bq + 3]
        return min(float(np.linalg.norm(self.data.site_xpos[s] - box_c))
                   for s in self.ids['palm_sites'])

    def _sync_weld(self, f):
        d, m = self.data, self.model
        eq = self.ids['weld_eq']
        ref = self.motion.qpos[f]
        if self.motion.meta_at(f)[3]:
            if not self.attached:
                if self._palm_box_dist() > self.args.engage_dist:
                    return
                self.attached = True
                d.eq_active[eq] = 1
                print(f'[{self.t:6.2f}s] weld ENGAGED '
                      f'(palm-box {self._palm_box_dist():.2f} m)')
            m.eq_data[eq, 0:3] = 0.0
            m.eq_data[eq, 3:6] = MQ.inv_mul_vec(ref[3:7],
                                                ref[36:39] - ref[0:3])
            m.eq_data[eq, 6:10] = MQ.mul(MQ.inv(ref[3:7]), ref[39:43])
        elif self.attached:
            self.attached = False
            d.eq_active[eq] = 0
            print(f'[{self.t:6.2f}s] weld RELEASED')

    def _check_fall(self):
        down = np.array([0.0, 0.0, -1.0])
        quat_wxyz = self.data.qpos[self.rq + 3:self.rq + 7]
        q_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3],
                           quat_wxyz[0]])
        gravity = quat_rotate_xyzw(quat_conj_xyzw(q_xyzw), down)
        root_z = float(self.data.qpos[self.rq + 2])
        if not self.fallen and (root_z < 0.30 or gravity[2] > 0.1):
            self.fallen, self.fall_time = True, self.t
            print(f'[{self.t:6.2f}s] ROBOT FELL')

    def _open_outputs(self):
        a = self.args
        self.renderer = self.writer = self.cam = None
        self.look = np.array([0.8, 0.0, 0.7])
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
        f = min(self.frame, self.motion.timesteps - 1)
        gq = self.motion.qpos[f]
        g = self.gdata
        g.qpos[:] = 0.0
        g.qpos[self.rq + 3] = 1.0
        g.qpos[self.rq:self.rq + 7] = gq[0:7]
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
        if self.mode != 'kinematic' and scn.ngeom < scn.maxgeom:
            gm = scn.geoms[scn.ngeom]
            mat = np.empty(9)
            mujoco.mju_quat2Mat(mat, np.asarray(gq[39:43], float))
            R = mat.reshape(3, 3) @ box_scene.GHOST_MAT
            pos = gq[36:39] + mat.reshape(3, 3) @ box_scene.GHOST_CENTER
            mujoco.mjv_initGeom(gm, mujoco.mjtGeom.mjGEOM_BOX,
                                box_scene.GHOST_HALF,
                                np.asarray(pos, float), R.ravel(), GHOST_RGBA)
            scn.ngeom += 1

    def render(self):
        if self.renderer is None:
            return
        box_xy = self.data.qpos[self.bq:self.bq + 2]
        focus = np.array([0.55 * self.data.qpos[self.rq] + 0.45 * box_xy[0],
                          0.55 * self.data.qpos[self.rq + 1]
                          + 0.45 * box_xy[1], 0.7])
        self.look += 0.06 * (focus - self.look)
        self.cam.lookat[:] = self.look
        self.renderer.update_scene(self.data, camera=self.cam)
        self._draw_ghost(self.renderer.scene)
        self.writer.append_data(self.renderer.render())

    def _robot_box_dist(self):
        return float(np.linalg.norm(self.data.qpos[self.rq:self.rq + 2]
                                    - self.data.qpos[self.bq:self.bq + 2]))

    def run(self):
        a = self.args
        wall = time.time()
        for tick in range(int(a.max_seconds * POLICY_FPS)):
            self.t = tick * P.CONTROL_DT
            self.step_control()
            self._check_fall()
            self.render()
            if self.viewer is not None:
                if not self.viewer.is_running():
                    break
                scn = getattr(self.viewer, 'user_scn', None)
                if scn is not None:
                    scn.ngeom = 0
                    self._draw_ghost(scn)
                self.viewer.sync()
            if self.fallen and (self.t - self.fall_time) > 2.0:
                break
            if self.place_done and self._robot_box_dist() > self.WALK_AWAY_DIST:
                break
            if tick % 100 == 0:
                f = min(self.frame, self.motion.timesteps - 1)
                print(f'  t={self.t:5.1f}s mm={self.matcher.state_name():10s} '
                      f'x={self.data.qpos[self.rq]:5.2f} '
                      f'box_z={self.data.qpos[self.bq + 2]:5.2f} '
                      f'xy_err={np.linalg.norm(self.data.qpos[self.rq:self.rq + 2] - self.motion.qpos[f][0:2]):.2f}',
                      flush=True)
        return self.finish(wall)

    def finish(self, wall):
        if self.writer is not None:
            self.writer.close()
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
        d = self.data
        away = self._robot_box_dist()
        # transient lag up to ~0.75 m happens where the OmniRetarget lift
        # translates fast (out of the SceneBot training distribution); only
        # a runaway counts as lost tracking
        tracked = self.max_xy_err < 0.9
        squatted = self.min_root_z < 0.60
        lifted = (self.mode == 'kinematic'
                  or self.max_box_z > C.BOX_REST_Z + 0.25)
        ok = (self.pick_triggered and self.place_done and not self.fallen
              and tracked and squatted and lifted and away > 1.0)
        print(f'[mm-scenebot:{self.mode}] {"SUCCESS" if ok else "INCOMPLETE"} -- '
              f'{self.t:.1f} s sim, {time.time() - wall:.0f} s wall, '
              f'picked={self.pick_triggered}, placed={self.place_done}, '
              f'fallen={self.fallen}, max xy err {self.max_xy_err:.2f} m, '
              f'min root z {self.min_root_z:.2f} m, '
              f'max box z {self.max_box_z:.2f} m, robot-box dist {away:.2f} m')
        if self.writer is not None:
            print(f'[mm-scenebot] wrote {self.args.video_path}')
        return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['kinematic', 'weld', 'grasp'],
                    default='kinematic')
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--viewer', action='store_true')
    ap.add_argument('--max-seconds', type=float, default=40.0)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--settle-seconds', type=float, default=1.5)
    ap.add_argument('--carry-seconds', type=float, default=3.0)
    ap.add_argument('--post-place-pause', type=float, default=2.0)
    ap.add_argument('--arm-gain', type=float, default=6.0)
    ap.add_argument('--squeeze', type=float, default=0.4)
    ap.add_argument('--engage-dist', type=float, default=0.45)
    ap.add_argument('--box-mass', type=float, default=0.5)
    ap.add_argument('--video',
                    default=os.path.join(HERE, 'out', 'scenebot_mm_pickup.mp4'))
    args = ap.parse_args()
    args.video_path = args.video
    return Demo(args).run()


if __name__ == '__main__':
    sys.exit(main())
