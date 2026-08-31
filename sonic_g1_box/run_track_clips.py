"""Track each OmniRetarget box clip directly with SONIC -- no motion matching.

Robot + box initialize at the clip's first frame (the start of the pickup);
the clip itself (resampled 30 -> 50 Hz) is the reference SONIC tracks in full
physics with the frictional carton. The label-gated squeeze/open biases work
exactly as in run_grasp, driven by the clip's own labels (sidecar or rule).

    python run_track_clips.py                        # all clips x all variants, headless
    python run_track_clips.py --video   # + out/track_clips/<stem>_<sonic>_<success|fail>.mp4
    python run_track_clips.py --viewer --sonic release --clips sub12_largebox_071_original_mujoco
"""
import argparse
import os
import sys

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

from mm_g1 import config as C
from mm_g1 import labels
from mm_g1.data import _box_clip_names, _load_box_npz
from mm_g1.states import Phase

import box_scene
from demo_base import GHOST_RGBA, CONTACT_RGBA, _EYE3
from mm_stream import nlerp
from run_grasp import (L_SHOULDER_ROLL, R_SHOULDER_ROLL,
                       L_WRIST_YAW, R_WRIST_YAW, CUSTOM_ARM_JOINTS)

FALL_Z = 0.28
POLICY_FPS = 1.0 / P.CONTROL_DT


class ClipMotion:
    """One clip resampled to 50 Hz, exposing the interface SonicPolicy reads."""

    def __init__(self, robot_q, box_pose):
        q30 = np.concatenate([robot_q, box_pose], axis=1)        # (T30, 43)
        T30 = len(q30)
        t50 = np.arange(0, (T30 - 1) / C.FPS, P.CONTROL_DT)
        i0 = np.minimum((t50 * C.FPS).astype(int), T30 - 2)
        a = (t50 * C.FPS - i0)[:, None]
        q = (1 - a) * q30[i0] + a * q30[i0 + 1]                  # (T50, 43)
        for k, (lo, hi) in enumerate(((3, 7), (39, 43))):
            q[:, lo:hi] = np.array([nlerp(q30[j, lo:hi], q30[j + 1, lo:hi], a[m, 0])
                                    for m, j in enumerate(i0)])
        self.qpos = q
        self.timesteps = len(q)
        self.src30 = i0                                          # 50 Hz -> 30 Hz frame
        ang = q[:, 7:36][:, P.MUJOCO_TO_ISAACLAB]                # (T50, 29)
        vel = np.zeros_like(ang)
        vel[1:] = (ang[1:] - ang[:-1]) * POLICY_FPS
        self.joint_pos, self.joint_vel = ang, vel
        self.body_pos = q[:, None, 0:3]
        self.body_quat = q[:, None, 3:7]


class _Render:
    def __init__(self, model, data, bq, q_at, ghost, path=None, viewer=False):
        self.model, self.data, self.bq, self.q_at = model, data, bq, q_at
        self.ghost = ghost                       # (centre, axes, half) box OBB
        self.gdata = mujoco.MjData(model)        # ghost FK only
        self.gbodies = [b for b in range(1, model.nbody)
                        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
                        != 'largebox']
        self.writer = self.renderer = self.viewer = None
        self.look = self._focus()
        if path:
            import imageio
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.renderer = mujoco.Renderer(model, 720, 1280)
            self.writer = imageio.get_writer(path, fps=int(POLICY_FPS),
                                             codec='libx264', quality=8,
                                             macro_block_size=None)
            self.cam = mujoco.MjvCamera()
            self.cam.distance, self.cam.azimuth, self.cam.elevation = \
                2.8, 120.0, -18.0
        if viewer:
            import mujoco.viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(
                model, data, show_left_ui=False, show_right_ui=False)
            self.viewer.cam.distance, self.viewer.cam.azimuth = 2.8, 120.0
            self.viewer.cam.elevation = -18.0
            self.viewer.cam.lookat[:] = self.look

    def _focus(self):
        d = self.data
        return np.array([0.55 * d.qpos[0] + 0.45 * d.qpos[self.bq],
                         0.55 * d.qpos[1] + 0.45 * d.qpos[self.bq + 1], 0.7])

    def _draw_ghost(self, scn, ref_q, in_contact):
        """Amber reference skeleton + reference box (green while a hand-contact
        label is on) -- the same overlay as the grasp demo."""
        g = self.gdata
        g.qpos[:] = 0.0
        g.qpos[3] = 1.0
        g.qpos[0:7] = ref_q[0:7]
        g.qpos[self.q_at] = ref_q[7:36]
        mujoco.mj_kinematics(self.model, g)
        for b in self.gbodies:
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
        if scn.ngeom < scn.maxgeom:
            gc, gmat, ghalf = self.ghost
            gm = scn.geoms[scn.ngeom]
            mat = np.empty(9)
            mujoco.mju_quat2Mat(mat, np.asarray(ref_q[39:43], float))
            R = mat.reshape(3, 3) @ gmat
            pos = ref_q[36:39] + mat.reshape(3, 3) @ gc
            mujoco.mjv_initGeom(gm, mujoco.mjtGeom.mjGEOM_BOX, ghalf,
                                np.asarray(pos, float), R.ravel(),
                                CONTACT_RGBA if in_contact else GHOST_RGBA)
            scn.ngeom += 1

    def frame(self, ref_q, in_contact):
        self.look += 0.06 * (self._focus() - self.look)
        if self.renderer is not None:
            self.cam.lookat[:] = self.look
            self.renderer.update_scene(self.data, camera=self.cam)
            self._draw_ghost(self.renderer.scene, ref_q, in_contact)
            self.writer.append_data(self.renderer.render())
        if self.viewer is not None:
            import time
            scn = getattr(self.viewer, 'user_scn', None)
            if scn is not None:
                scn.ngeom = 0
                self._draw_ghost(scn, ref_q, in_contact)
            self.viewer.cam.lookat[:] = self.look
            self.viewer.sync()
            time.sleep(P.CONTROL_DT)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.viewer is not None:
            self.viewer.close()


def track_clip(stem, variant, args):
    robot_q, box_pose = _load_box_npz(stem)
    phase, _, contact = labels.box_labels(stem, box_pose, C.BOX_DATA_DIR)
    motion = ClipMotion(robot_q, box_pose)
    z_rest = float(np.median(box_pose[:C.BOX_REST_FRAMES, 2]))
    ref_peak = float(box_pose[:, 2].max())

    scene_xml = os.path.join(ROOT, 'assets', 'scenebot', 'scene_robot_only.xml')
    model, ids = box_scene.build_model(scene_xml, 'grasp', box_mass=args.box_mass,
                                       box_type='carton',
                                       box_friction=args.box_friction)
    model.opt.timestep = P.CONTROL_DT / args.substeps
    data = mujoco.MjData(model)
    q_at = np.array([model.joint(n).qposadr[0] for n in P.JOINT_NAMES_MUJOCO])
    dq_at = np.array([model.joint(n).dofadr[0] for n in P.JOINT_NAMES_MUJOCO])
    bq = ids['box_qpos_at']
    kps, kds = P.KPS.copy(), P.KDS.copy()
    kps[CUSTOM_ARM_JOINTS] *= args.arm_kp
    kds[CUSTOM_ARM_JOINTS] *= args.arm_kd

    q0 = motion.qpos[0]
    data.qpos[0:7] = q0[0:7]
    data.qpos[q_at] = q0[7:36]
    data.qpos[bq:bq + 7] = q0[36:43]
    mujoco.mj_forward(model, data)
    # Feet flush with the floor: the clip's root z is data, not this scene.
    foot_lo = min(float(data.geom_xpos[g][2]) - float(model.geom_size[g][2])
                  for g in range(model.ngeom)
                  if 'foot' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''))
    data.qpos[2] -= foot_lo - 0.002
    mujoco.mj_forward(model, data)

    policy = SonicPolicy(variant=variant, device='cpu')
    policy.streaming = True                       # hold the last frame at the end
    policy.set_motion(motion)
    # Shared world frame: pin heading alignment to identity (as in demo_base).
    policy.heading_init_base_quat = np.array([1.0, 0.0, 0.0, 0.0])
    policy.init_ref_root_quat = np.array([1.0, 0.0, 0.0, 0.0])
    policy.delta_heading = 0.0
    policy.reinitialize_heading = False
    policy._update_heading = lambda base_quat: None

    def tick():
        # freejoint qvel[3:6] is already body-local (gyro convention)
        target = policy.step(data.qpos[3:7].copy(), data.qvel[3:6].copy(),
                             data.qpos[q_at], data.qvel[dq_at])
        f50 = min(policy.current_frame, motion.timesteps - 1)
        f30 = motion.src30[f50]
        lc, rc = contact[f30, 2] > 0.5, contact[f30, 3] > 0.5
        tick.f50, tick.in_contact = f50, bool(lc or rc)
        if lc or rc:
            target = target.copy()
            target[L_SHOULDER_ROLL] -= args.shoulder_squeeze
            target[R_SHOULDER_ROLL] += args.shoulder_squeeze
            target[L_WRIST_YAW] -= args.wrist_squeeze
            target[R_WRIST_YAW] += args.wrist_squeeze
        elif phase[f30] in (Phase.PICK, Phase.PLACE):
            target = target.copy()
            target[L_SHOULDER_ROLL] += args.shoulder_open
            target[R_SHOULDER_ROLL] -= args.shoulder_open
        for _ in range(args.substeps):
            data.ctrl[:29] = (kps * (target - data.qpos[q_at])
                              - kds * data.qvel[dq_at])
            mujoco.mj_step(model, data)

    tmp = (os.path.join(HERE, 'out', 'track_clips', f'.{variant}_{stem}.tmp.mp4')
           if args.video else None)
    ren = _Render(model, data, bq, q_at, ids['ghost'], path=tmp,
                  viewer=args.viewer)
    # Success = still holding when the reference starts the put-down.
    ref_z = motion.qpos[:, 38]
    carry = np.flatnonzero(ref_z > z_rest + 0.25)
    carry_end = int(carry[-1]) if len(carry) else motion.timesteps - 1
    z_at_carry_end = None
    for _ in range(int(args.settle / P.CONTROL_DT)):     # policy paused at frame 0
        tick()
        ren.frame(motion.qpos[tick.f50], tick.in_contact)
    policy.start_play()
    peak, fallen, fall_f = 0.0, False, None
    for _ in range(motion.timesteps + int(1.0 / P.CONTROL_DT)):
        tick()
        ren.frame(motion.qpos[tick.f50], tick.in_contact)
        peak = max(peak, float(data.qpos[bq + 2]))
        if z_at_carry_end is None and tick.f50 >= carry_end:
            z_at_carry_end = float(data.qpos[bq + 2])
        if data.qpos[2] < FALL_Z:
            fallen, fall_f = True, int(policy.current_frame)
            break
    ren.close()
    lifted = peak > z_rest + 0.25
    success = (lifted and not fallen and z_at_carry_end is not None
               and z_at_carry_end > z_rest + 0.15)
    if tmp:
        final = os.path.join(HERE, 'out', 'track_clips',
                             f'{stem}_{variant}_'
                             f'{"success" if success else "fail"}.mp4')
        os.replace(tmp, final)
    return dict(lifted=lifted, peak=peak, ref_peak=ref_peak, success=success,
                end_z=float(data.qpos[bq + 2]), fallen=fallen, fall_f=fall_f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', nargs='*', default=None)
    ap.add_argument('--sonic', nargs='+', default=['release', 'sonic_v1_1'],
                    choices=list(P.SONIC_VARIANTS))
    ap.add_argument('--shoulder-squeeze', type=float, default=0.6)
    ap.add_argument('--wrist-squeeze', type=float, default=0.2)
    ap.add_argument('--shoulder-open', type=float, default=0.4)
    ap.add_argument('--arm-kp', type=float, default=1.5)
    ap.add_argument('--arm-kd', type=float, default=1.0)
    ap.add_argument('--box-mass', type=float, default=0.25)
    ap.add_argument('--box-friction', type=float, default=1.5)
    ap.add_argument('--substeps', type=int, default=20)
    ap.add_argument('--settle', type=float, default=1.0)
    ap.add_argument('--video', action='store_true',
                    help='write out/track_clips/<variant>_<stem>.mp4 per run')
    ap.add_argument('--viewer', action='store_true',
                    help='watch live (one window per clip; needs a display)')
    args = ap.parse_args()
    stems = args.clips or _box_clip_names()

    for variant in args.sonic:
        print(f'\n=== {variant} ({len(stems)} clips) ===')
        wins = 0
        for stem in stems:
            r = track_clip(stem, variant, args)
            wins += r['success']
            status = (f"FELL@{r['fall_f']}" if r['fallen'] else
                      'SUCCESS' if r['success'] else
                      'dropped' if r['lifted'] else 'slipped')
            print(f'  {stem:42s} {status:8s} peak {r["peak"]:.2f} '
                  f'(ref {r["ref_peak"]:.2f}) end z {r["end_z"]:.2f}')
        print(f'  -> {wins}/{len(stems)} carried without dropping')


if __name__ == '__main__':
    main()
