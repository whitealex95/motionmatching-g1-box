"""Retarget the EMM box-carry take to the G1 with OmniRetargeting, shrinking the box.

The EMM `box.bvh` actor carries a ~0.52 m wide box for the whole take, but the box was
never exported: the hands are ~0.6 m apart and the G1 (GMR retarget in
data/emm_g1/clips/emm_extra__box.npz) holds its hands just as wide, far wider than the
0.3 x 0.2 x 0.3 m box the pick skill grabs with its wrists 0.42 m apart.

This runner builds the carried box from the hands (centre at the wrist midpoint, one
axis along the wrist-to-wrist line, box width = wrist separation minus the human hand
clearance) and feeds OmniRetargeting TWO point sets per frame:

  object_points         the box as the actor carried it (source interaction mesh)
  target_object_points  the same lattice shrunk to the robot's box (robot mesh locks it)

The Laplacian coordinates of the human mesh encode "hands just outside the box faces",
so the robot's wrists close in on the smaller box while the rest of the body follows
the actor. Requires the `emm-box` branch of ~/Projects/omniretargeting (the
`target_object_points` patch) in the `omniretargeting` conda env:

    conda activate omniretargeting
    python tools/omniretarget_emm/retarget_emm_box.py --start-sec 10 --end-sec 20 \
        --out /tmp/test.npz --video /tmp/test.mp4

Output npz: qpos (T, 36) float32 in the g1.xml layout [root pos, root quat wxyz, 29 dof],
fps, source, box_pose (T, 7) target box centre + quat wxyz, and the run settings.
"""
import argparse
import json
import os
import tempfile
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import omniretargeting
from omniretargeting import OmniRetargeter
from omniretargeting.data_sources.base import MotionData
from omniretargeting.data_sources.lafan1 import _quat_fk, _read_bvh
from omniretargeting.robot_config import load_robot_config
from omniretargeting.utils import create_flat_terrain, estimate_body_height

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OMNI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(omniretargeting.__file__)))
G1_PROFILE = os.path.join(OMNI_ROOT, "robot_models", "unitree_g1", "unitree_g1.json")
EMM_BVH = os.path.expanduser(
    "~/Projects/Environment-aware-Motion-Matching/DataEMM/emm_extra/box.bvh")

# BVH (Y-up, metres, character faces +Z) -> Z-up with the character facing +X.
_Y_TO_Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
_FWD = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
BVH_TO_WORLD = _FWD @ _Y_TO_Z

# EMM skeleton -> G1 links (same link choices as the upstream lafan1 profile).
EMM_MAPPING = {
    "Hips":          ("pelvis", (0, 0, 0)),
    "LeftHip":       ("left_hip_roll_link", (0, 0, 0)),
    "RightHip":      ("right_hip_roll_link", (0, 0, 0)),
    "LeftKnee":      ("left_knee_link", (0, 0, 0)),
    "RightKnee":     ("right_knee_link", (0, 0, 0)),
    "LeftAnkle":     ("left_ankle_roll_link", (0, 0, 0.03)),
    "RightAnkle":    ("right_ankle_roll_link", (0, 0, 0.03)),
    "LeftToe":       ("left_ankle_roll_link", (0.12, 0, 0)),
    "RightToe":      ("right_ankle_roll_link", (0.12, 0, 0)),
    "Chest":         ("waist_roll_link", (0, 0, 0)),
    "LeftShoulder":  ("left_shoulder_roll_link", (0, 0, 0)),
    "RightShoulder": ("right_shoulder_roll_link", (0, 0, 0)),
    "LeftElbow":     ("left_elbow_link", (0, 0, 0)),
    "RightElbow":    ("right_elbow_link", (0, 0, 0)),
    "LeftWrist":     ("left_wrist_yaw_link", (0, 0, 0)),
    "RightWrist":    ("right_wrist_yaw_link", (0, 0, 0)),
}
EMM_BASE_ORIENTATION = {"pelvis": "Hips", "left_hip": "LeftHip",
                        "right_hip": "RightHip", "spine": "Chest2"}

# Virtual hand targets that make the palms FACE the box: a palm-centre point and a
# fingertip point on each hand, expressed in the robot's wrist_yaw_link frame (G1 rubber
# hand: fingers along +x, 0.04..0.17; palm surface at y = -0.02 on the left hand and
# +0.02 on the right, the thumb side). The source counterparts are the same offsets
# in the BOX frame (x forward, y right->left, z up) so matching them aligns the hand
# frame with the box frame: fingers forward along the side face, palm on it.
HAND_PALM_X, HAND_PALM_Y, HAND_FINGER_X = 0.10, 0.02, 0.17
HAND_TARGETS = {
    "LeftPalm":    ("LeftWrist", "left_wrist_yaw_link", (HAND_PALM_X, -HAND_PALM_Y, 0.0)),
    "RightPalm":   ("RightWrist", "right_wrist_yaw_link", (HAND_PALM_X, HAND_PALM_Y, 0.0)),
    "LeftFinger":  ("LeftWrist", "left_wrist_yaw_link", (HAND_FINGER_X, 0.0, 0.0)),
    "RightFinger": ("RightWrist", "right_wrist_yaw_link", (HAND_FINGER_X, 0.0, 0.0)),
}

# 3x3x3 lattice on the box surface (centre dropped): 26 points per frame.
_u = np.array([-1.0, 0.0, 1.0])
BOX_LATTICE = np.array([[x, y, z] for x in _u for y in _u for z in _u
                        if not (x == 0 and y == 0 and z == 0)])   # (26, 3)


def load_bvh_world(path, start, stop):
    """(T, J, 3) world joint positions (Z-up, metres) and the joint names."""
    bvh = _read_bvh(path)
    _, gpos = _quat_fk(bvh["quats"][start:stop], bvh["positions"][start:stop], bvh["parents"])
    return gpos.astype(np.float64) @ BVH_TO_WORLD.T, list(bvh["names"]), 1.0 / bvh["frametime"]


def box_frames(lw, rw):
    """Per-frame box rotation (T, 3, 3) with x forward, y along right->left wrist, z up."""
    y = lw - rw
    y[:, 2] = 0.0
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    z = np.tile([0.0, 0.0, 1.0], (len(y), 1))
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=-1)                               # (T, 3, 3) columns


def lattice_points(center, rot, half):
    """(T, 26, 3) world points of the lattice for per-frame half extents (T, 3)."""
    local = BOX_LATTICE[None] * half[:, None, :]                      # (T, 26, 3)
    return np.einsum("tij,tnj->tni", rot, local) + center[:, None, :]


def hand_target_points(pos, names, rot):
    """(T, 4, 3) virtual palm / fingertip points: the robot offsets laid out in the box frame."""
    j = {n: k for k, n in enumerate(names)}
    out = []
    for wrist, _, off in HAND_TARGETS.values():
        out.append(pos[:, j[wrist]] + np.einsum("tij,j->ti", rot, np.asarray(off)))
    return np.stack(out, 1)


def build_retargeter(terrain_path, retargeting_cfg, source_names, hand_targets=True):
    mapping = {src: {"robot_link": link, "offset": list(off)}
               for src, (link, off) in EMM_MAPPING.items()}
    if hand_targets:
        mapping.update({src: {"robot_link": link, "offset": list(off)}
                        for src, (_, link, off) in HAND_TARGETS.items()})
    return OmniRetargeter(
        robot_urdf_path=os.path.join(OMNI_ROOT, "robot_models", "unitree_g1", "g1_29dof_popsicle.urdf"),
        terrain_mesh_path=terrain_path,
        joint_mapping=mapping,
        source_target_names=source_names,   # all BVH joints: positions are indexed by these
        base_orientation=EMM_BASE_ORIENTATION,
        retargeting=retargeting_cfg,
    )


def fk_wrists(model, data, qpos):
    import mujoco
    ids = [model.body("left_wrist_yaw_link").id, model.body("right_wrist_yaw_link").id,
           model.body("pelvis").id]
    out = np.zeros((len(qpos), 3, 3))
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        out[t] = data.xpos[ids]
    return out


def hand_facing(model, data, qpos, box_rot):
    """Mean cosines of (palm normal, towards box) and (finger axis, box forward) per hand."""
    import mujoco
    ids = [model.body("left_wrist_yaw_link").id, model.body("right_wrist_yaw_link").id]
    palm_sign = (-1.0, 1.0)                     # palm normal in the link frame: -y left, +y right
    toward = (-1.0, 1.0)                        # box centre lies at -y_box of the left hand, +y_box of the right
    acc = np.zeros(4)
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        for k in range(2):
            Rw = data.xmat[ids[k]].reshape(3, 3)
            acc[k] += np.dot(palm_sign[k] * Rw[:, 1], toward[k] * box_rot[t][:, 1])
            acc[2 + k] += np.dot(Rw[:, 0], box_rot[t][:, 0])
    return acc / len(qpos)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bvh", default=EMM_BVH)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=None)
    ap.add_argument("--fps", type=float, default=30.0, help="output frame rate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", default=None, help="render the result (needs MUJOCO_GL=egl)")
    ap.add_argument("--wrist-sep", type=float, default=0.42,
                    help="robot wrist-yaw-link separation to aim for (the SceneBot hold: 0.42 m)")
    ap.add_argument("--hand-clearance", type=float, default=0.04,
                    help="human wrist-joint to box-face distance (m, source scale)")
    ap.add_argument("--source-box", type=float, nargs=2, default=(0.32, 0.30),
                    help="actor's box depth (fore-aft) and height (m); width comes from the hands")
    ap.add_argument("--target-box", type=float, nargs=3, default=(0.30, 0.20, 0.30),
                    help="the robot's real box x y z (m), x forward, y across the hands")
    ap.add_argument("--hold-offset", type=float, nargs=2, default=None, metavar=("X", "Z"),
                    help="place the target box at this forward / up offset from the pelvis "
                         "(m, heading frame) instead of where the actor held it; the SceneBot "
                         "pick leaves it at 0.265 0.06")
    ap.add_argument("--no-box", action="store_true", help="plain retarget, no object in the mesh")
    ap.add_argument("--no-hand-targets", action="store_true",
                    help="drop the virtual palm / fingertip targets (hands then face anywhere)")
    ap.add_argument("--penetration-resolver", default=None,
                    choices=("hard_constraint", "hard_constraint_slack", "xyz_nudge"),
                    help="override the G1 profile's resolver (xyz_nudge adds batch foot stabilization)")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    bvh = _read_bvh(args.bvh)
    src_fps = 1.0 / bvh["frametime"]
    n_src = len(bvh["quats"])
    start = int(round(args.start_sec * src_fps))
    stop = n_src if args.end_sec is None else min(n_src, int(round(args.end_sec * src_fps)))
    pos, names, _ = load_bvh_world(args.bvh, start, stop)             # (T, J, 3)
    T = len(pos)
    print(f"BVH frames {start}..{stop} ({T} @ {src_fps:.0f} fps)")

    human_height = estimate_body_height(pos, names, head_joint="Head",
                                        foot_joints=("LeftToe", "RightToe"))
    profile = load_robot_config(G1_PROFILE)
    terrain = create_flat_terrain(size=40.0)
    fd, terrain_path = tempfile.mkstemp(suffix=".obj")
    os.close(fd)
    terrain.export(terrain_path)
    retargeting_cfg = dict(profile.get("retargeting") or {})
    if args.penetration_resolver:
        retargeting_cfg["penetration_resolver"] = args.penetration_resolver
    hand_targets = not args.no_hand_targets
    all_names = names + (list(HAND_TARGETS) if hand_targets else [])
    retargeter = build_retargeter(terrain_path, retargeting_cfg, all_names, hand_targets)
    scale = retargeter.robot_height / human_height
    print(f"human height {human_height:.3f} m, robot {retargeter.robot_height:.3f} m -> scale {scale:.4f}")

    pos = pos * scale                                                 # robot-sized actor
    j = {n: k for k, n in enumerate(names)}
    lw, rw = pos[:, j["LeftWrist"]], pos[:, j["RightWrist"]]
    sep = np.linalg.norm(lw - rw, axis=1)                             # (T,)
    center = 0.5 * (lw + rw)
    rot = box_frames(lw, rw)
    if args.hold_offset is not None:
        # Edit the ACTOR: carry the box (and the hands on it) at the requested forward /
        # height offset from the pelvis, keeping the lateral offset and yaw. The source
        # mesh then has hands-on-box consistent with where the robot must hold it.
        pelvis = pos[:, j["Hips"]]
        rel = np.einsum("tji,tj->ti", rot, center - pelvis)           # (T, 3) in the box frame
        print(f"actor held the box at x {rel[:, 0].mean():.3f} z {(center[:, 2] - pelvis[:, 2]).mean():.3f} "
              f"from the pelvis; moving it to {args.hold_offset}")
        rel[:, 0] = args.hold_offset[0]
        moved = pelvis + np.einsum("tij,tj->ti", rot, rel)
        moved[:, 2] = pelvis[:, 2] + args.hold_offset[1]
        delta = moved - center                                        # (T, 3)
        pos[:, j["LeftWrist"]] += delta
        pos[:, j["RightWrist"]] += delta
        lw, rw = pos[:, j["LeftWrist"]], pos[:, j["RightWrist"]]
        center = moved
    if hand_targets:
        pos = np.concatenate([pos, hand_target_points(pos, names, rot)], 1)   # (T, J+4, 3)
    clearance = args.hand_clearance * scale
    src_half = np.stack([np.full(T, 0.5 * args.source_box[0] * scale),
                         0.5 * (sep - 2 * clearance),
                         np.full(T, 0.5 * args.source_box[1] * scale)], 1)   # (T, 3)
    tgt_half = np.tile([0.5 * args.target_box[0],
                        0.5 * (args.wrist_sep - 2 * clearance),
                        0.5 * args.target_box[2]], (T, 1))
    print(f"wrist separation {sep.mean():.3f} m (scaled) -> source box width "
          f"{2 * src_half[:, 1].mean():.3f}, target lattice width {2 * tgt_half[0, 1]:.3f}")

    motion = MotionData(
        positions=pos,
        target_names=all_names,
        root_translations=pos[:, j["Hips"]].copy(),
        framerate=src_fps,
        source_height=human_height * scale,
        object_points=None if args.no_box else lattice_points(center, rot, src_half),
        target_object_points=None if args.no_box else lattice_points(center, rot, tgt_half),
    )
    motion = motion.resample(args.fps)
    print(f"resampled to {len(motion.positions)} frames @ {args.fps:.0f} fps")

    t0 = time.time()
    _, out = retargeter.retarget_motion(motion, framerate=args.fps, visualize_trajectory=False,
                                        enable_scene_scaling=False, show_progress=args.progress)
    dt = time.time() - t0
    print(f"retargeted {len(out)} frames in {dt:.0f} s ({dt / len(out):.3f} s/frame)")

    names_out = retargeter.get_joint_names()
    qpos = out.astype(np.float32)                                      # (T, 36)
    lw_r, rw_r = motion.positions[:, j["LeftWrist"]], motion.positions[:, j["RightWrist"]]
    box_c = 0.5 * (lw_r + rw_r)
    box_q = R.from_matrix(box_frames(lw_r, rw_r)).as_quat(scalar_first=True)
    box_pose = np.concatenate([box_c, box_q], 1).astype(np.float32)   # (T, 7)

    fk = fk_wrists(retargeter.robot_model, retargeter.robot_data, out)   # (T, 3, 3)
    rsep = np.linalg.norm(fk[:, 0] - fk[:, 1], axis=1)
    print(f"robot wrist separation: mean {rsep.mean():.3f}  p5 {np.percentile(rsep, 5):.3f}  "
          f"p95 {np.percentile(rsep, 95):.3f}  (target {args.wrist_sep})")
    hand_err = np.linalg.norm(0.5 * (fk[:, 0] + fk[:, 1]) - box_c, axis=1)
    print(f"robot wrist-mid vs box centre: mean {hand_err.mean():.3f} m")
    facing = hand_facing(retargeter.robot_model, retargeter.robot_data, out, box_frames(lw_r, rw_r))
    print("hand facing (cosines, 1 = perfect): palm->box  L %.2f R %.2f   fingers forward  L %.2f R %.2f"
          % tuple(facing))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, qpos=qpos, fps=int(args.fps), source=args.bvh, dof_names=np.array(names_out),
             box_pose=box_pose, scale=scale, settings=json.dumps(vars(args)))
    print(f"saved {args.out}")
    os.remove(terrain_path)

    if args.video:
        from render_clip import render
        render(qpos, box_pose, args.video, fps=args.fps, box_size=args.target_box)


if __name__ == "__main__":
    main()
