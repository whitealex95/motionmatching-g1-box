"""EMM clips baked as walking-while-CARRY frames.

`emm_extra__box` is a 175.8 s take in which the actor carries a box for part of the
time, but the box itself was never captured: the npz is robot-only (36-D qpos), so
unlike the OmniRetarget clips there is no object channel. The carry is mimed, and the
hold is very stable -- over the extracted spans the wrist midpoint sits 0.24 m ahead of
the pelvis with a 1.6 cm standard deviation.

Only the carry spans are baked. They are found geometrically (both hands forward, level,
symmetric, a box-width apart, at waist height and NOT overhead -- the clip also contains
arm raises that pass every test but the height one), then each surviving span becomes its
own clip so a matched frame can never run out of the carry and into unrelated motion.

The box is synthesized at the wrist midpoint purely so the scene has something to draw:
the carry SEARCH is box-agnostic (features.build_db gives `carry` the same pose +
trajectory space as `loco`), so the box pose here never reaches the matching query.
"""
import os

import numpy as np
import mujoco

from . import config as C
from . import quat
from .states import Phase

WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def _wrists(qpos):
    """(T,2,3) world wrist positions by FK over the G1 model."""
    model = mujoco.MjModel.from_xml_path(C.SCENE_XML)
    data = mujoco.MjData(model)
    ids = [model.body(n).id for n in WRIST_BODIES]
    out = np.zeros((len(qpos), 2, 3))
    for t, row in enumerate(qpos):
        data.qpos[:36] = row
        mujoco.mj_kinematics(model, data)
        out[t] = data.xpos[ids]
    return out


def _yaw_of(q_wxyz):
    w, x, y, z = q_wxyz.T
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def carry_spans(qpos, wrists):
    """Frame ranges [(start, stop), ...] where both hands hold a box-width apart."""
    yaw = _yaw_of(qpos[:, 3:7])
    c, s = np.cos(-yaw), np.sin(-yaw)
    local = np.zeros_like(wrists)                              # (T,2,3) in the base frame
    for k in range(2):
        v = wrists[:, k] - qpos[:, 0:3]
        local[:, k] = np.stack([c * v[:, 0] - s * v[:, 1],
                                s * v[:, 0] + c * v[:, 1], v[:, 2]], -1)
    L, R = local[:, 0], local[:, 1]
    mid = 0.5 * (L + R)
    sep = np.linalg.norm(L - R, axis=1)
    ok = ((np.minimum(L[:, 0], R[:, 0]) > C.EMM_HOLD_MIN_FWD)
          & (np.abs(L[:, 2] - R[:, 2]) < C.EMM_HOLD_MAX_DZ)
          & (np.abs(L[:, 1] + R[:, 1]) < C.EMM_HOLD_MAX_ASYM)
          & (sep > C.EMM_HOLD_MIN_SEP) & (sep < C.EMM_HOLD_MAX_SEP)
          & (mid[:, 2] > C.EMM_HOLD_MIN_Z) & (mid[:, 2] < C.EMM_HOLD_MAX_Z))
    # Close short dropouts, then keep only spans long enough to be searchable
    # (SEARCH_TAIL frames of every clip are excluded from its KD-tree).
    g = C.EMM_HOLD_GAP
    ok = np.convolve(ok.astype(int), np.ones(2 * g + 1, int), "same") > g
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])
            if b - a >= C.EMM_MIN_SPAN]


def _box_pose(qpos, wrists):
    """(T,7) world box pose riding the wrist midpoint, yawed with the root.

    Draw-only: the carry search never sees it. Height is clamped so the box centre sits
    at least its own half-height above the floor.
    """
    mid = wrists.mean(axis=1)                                   # (T,3)
    pose = np.zeros((len(qpos), 7))
    pose[:, 0:3] = mid
    pose[:, 2] = np.maximum(mid[:, 2], C.BOX_REST_Z)
    yaw = _yaw_of(qpos[:, 3:7])
    pose[:, 3] = np.cos(0.5 * yaw)
    pose[:, 6] = np.sin(0.5 * yaw)
    return pose


def build():
    """Bake the configured EMM clips' carry spans.

    Returns a list of (name, qpos, box_pose, contact, attach, phase_code), the same
    tuple scenebot_pick.build() returns.
    """
    out = []
    for stem in C.EMM_CLIPS:
        path = os.path.join(C.EMM_DATA_DIR, stem + ".npz")
        qpos = np.load(path)["qpos"].astype(np.float64)         # (T,36)
        wrists = _wrists(qpos)
        box = _box_pose(qpos, wrists)
        for k, (a, b) in enumerate(carry_spans(qpos, wrists)):
            # contact=None: data.py derives it from `attach` (both wrists on the box).
            out.append((f"{stem}__carry{k}", qpos[a:b].copy(), box[a:b].copy(),
                        None, np.ones(b - a, bool), Phase.CARRY))
    return out
