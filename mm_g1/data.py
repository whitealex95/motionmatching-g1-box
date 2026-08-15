"""Build / load the G1 locomotion library from the GMR-retargeted LAFAN1 clips.

The library concatenates the walk, run and push-and-stumble clips into one continuous
array and precomputes the per-frame heading and FK foot positions that the feature
extractor needs. It is cached to data/motion_lib.npz so subsequent launches start
instantly. See data/gmr_lafan1_g1/README.md for the source pickle format.
"""
import os
import glob
import pickle
import numpy as np

from . import config as C
from . import boxes
from . import quat
from .g1_model import G1Model, csv_to_qpos, quat_wxyz_yaw

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])


def _gmr_rows(name, data_dir):
    """GMR pickle {root_pos (T,3), root_rot (T,4) xyzw, dof_pos (T,29)} -> (T,36) rows in
    the project's [xyz, quat_xyzw, 29 joints] layout (same as the old CSVs)."""
    with open(os.path.join(data_dir, name + ".pkl"), "rb") as f:
        d = pickle.load(f)
    return np.concatenate([d["root_pos"], d["root_rot"], d["dof_pos"]], axis=1)


def _load_clip(name, data_dir=C.DATA_DIR, trim=None):
    """Load a clip and apply its GenoView-matched absolute [start:stop] frame window
    (CLIP_TRIM). This drops T-pose lead-in/out off walk & run and, crucially, isolates the
    short stumble EVENT out of the otherwise-ordinary pushAndStumble clip."""
    rows = _gmr_rows(name, data_dir)
    s, e = trim if trim is not None else C.CLIP_TRIM.get(name, (0, len(rows)))
    rows = rows[s:min(e, len(rows))]
    return csv_to_qpos(rows)  # (T, 36) wxyz; csv_to_qpos reorders quat xyzw -> wxyz


def _box_clip_names(data_dir=C.BOX_DATA_DIR):
    """Stems of the robot-object .npz clips to load (config BOX_CLIPS or every file)."""
    if C.BOX_CLIPS != "all":
        return [c for c in C.BOX_CLIPS
                if os.path.exists(os.path.join(data_dir, c + ".npz"))]
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(data_dir, "*.npz")))


def _load_box_npz(name, data_dir=C.BOX_DATA_DIR):
    """OmniRetarget robot-object clip -> (robot_qpos (T,36) wxyz, box_pose (T,7) pos+wxyz).

    The exported .npz is already in MuJoCo qpos layout (43-D): robot [0:36] then the box
    freejoint [36:39] position + [39:43] quaternion (wxyz). No reordering needed."""
    d = np.load(os.path.join(data_dir, name + ".npz"))
    q = d["qpos"].astype(np.float64)
    return q[:, 0:36].copy(), q[:, 36:43].copy()


def _yaw_box_pose(box_pose, angle):
    """Copy of a box trajectory (T,7)=[pos, quat wxyz] with only its ORIENTATION yawed by
    `angle` about the world vertical through the box's own centre. The centre position (and
    hence the robot's grip / reach) is untouched: pre-multiplying by a world-Z yaw spins the box
    in place. Because that yaw commutes with the root-frame yaw used in features.build_db, the
    resulting box search feature is an exact `angle` rotation of the original's in the base frame
    -- giving the near-square box rotational-symmetry coverage (pick/carry/place it at any facing)."""
    dq = quat.from_angle_axis(np.asarray(angle), UP)          # world-vertical yaw (wxyz)
    out = box_pose.copy()
    out[:, 3:7] = quat.mul(dq, box_pose[:, 3:7])
    return out


def build_library(clips=None, out=C.LIB_PATH):
    """Concatenate clips into one array; precompute heading and FK foot positions."""
    clips = clips or C.CLIPS
    clips = [c for c in clips if os.path.exists(os.path.join(C.DATA_DIR, c + ".pkl"))]
    if not clips:
        raise FileNotFoundError(f"No clips found in {C.DATA_DIR}")

    model = G1Model()
    # Locomotion clips are each (GenoView-style) added twice: normal + L/R MIRRORED,
    # for symmetric left/right coverage. The robot-object (box) clips are NOT sagittally mirrored
    # (a mirror would also have to reflect the paired box trajectory), but each IS replicated
    # BOX_ROT_FOLDS times with the box's orientation yawed about its own centre (0/90/180/270 deg
    # at N=4) so the box can be picked/carried/placed at any facing (see _yaw_box_pose). Each
    # concrete entry is (name, robot_qpos, kind, box_pose) where box_pose is None for non-box clips.
    box_clips = _box_clip_names()
    loaded = []
    for name in clips:                               # locomotion (walk / run / stumble)
        q = _load_clip(name)
        loaded.append((name, q, "loco", None))
        if C.MIRROR:
            loaded.append((name + "_mirror", model.mirror_qpos(q), "loco", None))
    folds = max(1, C.BOX_ROT_FOLDS)
    for name in box_clips:                            # OmniRetarget pick/carry/place clips
        robot_q, box_pose = _load_box_npz(name)
        for k in range(folds):                        # N-fold box-orientation augmentation
            bp = box_pose if k == 0 else _yaw_box_pose(box_pose, k * 2.0 * np.pi / folds)
            tag = name if k == 0 else f"{name}_rot{k}"
            loaded.append((tag, robot_q, "box", bp))

    qpos, clip_id, frame_in_clip, lengths, names = [], [], [], [], []
    skill, box_pose_all, box_attach = [], [], []
    n_box = 0
    for cid, (name, q, kind, bpose) in enumerate(loaded):
        n = len(q)
        if kind == "box":
            sk, at, _info = boxes.segment_phases(bpose[:, 0:3])
            bp = bpose
            n_box += 1
        else:                                        # locomotion
            sk = np.zeros(n, np.int32)
            bp, at = np.tile(np.r_[0, 0, 0, IDENTITY_QUAT], (n, 1)), np.zeros(n, bool)
        qpos.append(q); skill.append(sk)
        box_pose_all.append(bp); box_attach.append(at)
        clip_id.append(np.full(n, cid))
        frame_in_clip.append(np.arange(n))
        lengths.append(n); names.append(name)
        tag = f", {kind}" if kind == "box" else ""
        print(f"  [{cid}] {name}: {n} frames{tag}")

    qpos = np.concatenate(qpos)
    feet = model.fk_feet(qpos)                       # (N, 2, 3) world
    yaw = quat_wxyz_yaw(qpos[:, 3:7])                 # (N,)

    np.savez_compressed(
        out,
        qpos=qpos.astype(np.float32),
        feet_world=feet.astype(np.float32),
        yaw=yaw.astype(np.float32),
        clip_id=np.concatenate(clip_id).astype(np.int32),
        frame_in_clip=np.concatenate(frame_in_clip).astype(np.int32),
        lengths=np.array(lengths, np.int32),
        clip_names=np.array(names),
        skill=np.concatenate(skill).astype(np.int32),
        box_pose=np.concatenate(box_pose_all).astype(np.float32),
        box_attach=np.concatenate(box_attach),
        lib_version=np.array(C.LIB_VERSION),
    )
    print(f"Saved library: {qpos.shape[0]} frames, {len(loaded)} clips "
          f"({n_box} box) -> {out}")
    return out


def load_library(path=C.LIB_PATH):
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        version = int(d["lib_version"]) if "lib_version" in d.files else 0
        if version != C.LIB_VERSION:
            print("Cache is stale (library v%d != v%d); rebuilding..." % (version, C.LIB_VERSION))
            os.remove(path)                          # rebuild with the current heuristics
    if not os.path.exists(path):
        build_library(out=path)
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    build_library()
