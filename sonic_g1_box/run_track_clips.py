"""Track each OmniRetarget box clip directly with SONIC -- no motion matching.

Robot + box initialize at the clip's first frame (the start of the pickup);
the clip itself (resampled 30 -> 50 Hz) is the reference SONIC tracks in full
physics with the frictional carton. The label-gated squeeze/open biases work
exactly as in run_grasp, driven by the clip's own labels (sidecar or rule).

    python run_track_clips.py                        # all clips x all variants
    python run_track_clips.py --sonic release --clips sub12_largebox_071_original_mujoco
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import mujoco

from sonic_tracking import params as P
from sonic_tracking.policy import SonicPolicy

from mm_g1 import config as C
from mm_g1 import labels
from mm_g1.data import _box_clip_names, _load_box_npz
from mm_g1.states import Phase

import box_scene
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
        f30 = motion.src30[min(policy.current_frame, motion.timesteps - 1)]
        lc, rc = contact[f30, 2] > 0.5, contact[f30, 3] > 0.5
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

    for _ in range(int(args.settle / P.CONTROL_DT)):     # policy paused at frame 0
        tick()
    policy.start_play()
    peak, fallen, fall_f = 0.0, False, None
    for _ in range(motion.timesteps + int(1.0 / P.CONTROL_DT)):
        tick()
        peak = max(peak, float(data.qpos[bq + 2]))
        if data.qpos[2] < FALL_Z:
            fallen, fall_f = True, int(policy.current_frame)
            break
    lifted = peak > z_rest + 0.25
    return dict(lifted=lifted, peak=peak, ref_peak=ref_peak,
                end_z=float(data.qpos[bq + 2]), fallen=fallen, fall_f=fall_f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', nargs='*', default=None)
    ap.add_argument('--sonic', nargs='+', default=['release', 'sonic_v1_1'],
                    choices=list(P.SONIC_VARIANTS))
    ap.add_argument('--shoulder-squeeze', type=float, default=0.6)
    ap.add_argument('--wrist-squeeze', type=float, default=0.0)
    ap.add_argument('--shoulder-open', type=float, default=0.4)
    ap.add_argument('--arm-kp', type=float, default=1.5)
    ap.add_argument('--arm-kd', type=float, default=1.0)
    ap.add_argument('--box-mass', type=float, default=0.25)
    ap.add_argument('--box-friction', type=float, default=1.5)
    ap.add_argument('--substeps', type=int, default=20)
    ap.add_argument('--settle', type=float, default=1.0)
    args = ap.parse_args()
    stems = args.clips or _box_clip_names()

    for variant in args.sonic:
        print(f'\n=== {variant} ({len(stems)} clips) ===')
        wins = 0
        for stem in stems:
            r = track_clip(stem, variant, args)
            wins += r['lifted'] and not r['fallen']
            status = (f"FELL@{r['fall_f']}" if r['fallen'] else
                      'LIFTED' if r['lifted'] else 'slipped')
            print(f'  {stem:42s} {status:8s} peak {r["peak"]:.2f} '
                  f'(ref {r["ref_peak"]:.2f}) end z {r["end_z"]:.2f}')
        print(f'  -> {wins}/{len(stems)} lifted without falling')


if __name__ == '__main__':
    main()
