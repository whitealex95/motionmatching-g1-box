"""SceneBot's squat pickup (clip 11) baked as the library's single pick / drop pair.

The web demo plays clip 11 frames 0..120 at HALF speed for the pickup and the
same frames BACKWARD at full speed for the put-down. Both playbacks are baked
here exactly as played (so the matcher matches the 2x-slow pick and the
reversed drop, not the raw clip), resampled from the demo's 50 Hz to the
library's 30 Hz, together with the demo's per-frame contact labels.

The clip has no box, so a box trajectory is synthesized: the demo's free box
(0.3 x 0.2 x 0.3 m) rests where the hands close (the wrist-midpoint minimum,
0.29 m ahead of the stance) and rides the wrist midpoint + root yaw from that
frame on. Reversing the samples gives the matching set-down trajectory.
"""
import os
import sys

import numpy as np

from . import config as C
from .states import Phase
from . import quat

_TOOLS = os.path.join(C.ROOT, 'tools')
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from scenebot_tracking import params as SP                 # noqa: E402
from scenebot_tracking.clips import ClipBundle, ContactLabels  # noqa: E402

WRIST_BODIES = (28, 29)          # Isaac BFS: L / R wrist yaw links
UP = np.array([0.0, 0.0, 1.0])


def _nlerp(a, b, t):
    if np.dot(a, b) < 0.0:
        b = -b
    q = (1.0 - t) * a + t * b
    return q / (np.linalg.norm(q) + 1e-12)


def _yaw_of(q_wxyz):
    w, x, y, z = q_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class _Sampler:
    """Fractional-frame access to clip 11 (frames lo..hi) + its contact rows."""

    def __init__(self):
        bundle = ClipBundle(SP.CLIPS_BIN, SP.CLIPS_INDEX)
        contacts = ContactLabels(SP.CONTACT_BIN, SP.CONTACT_INDEX)
        clip = bundle.clip(C.SCENEBOT_CLIP)
        lo, hi = C.SCENEBOT_FRAMES
        self.lo, self.hi = lo, hi
        self.jp = clip.joint_pos[lo:hi + 1].astype(np.float64)      # (F, 29) isaac
        self.root_p = clip.body_pos[lo:hi + 1, 0].astype(np.float64)   # (F, 3)
        self.root_q = clip.body_quat[lo:hi + 1, 0].astype(np.float64)  # (F, 4) wxyz
        self.hand_mid = clip.body_pos[lo:hi + 1, WRIST_BODIES].mean(1).astype(np.float64)  # (F, 3)
        self.labels = np.array([contacts.at_frame(C.SCENEBOT_CLIP, lo + f)
                                for f in range(hi - lo + 1)], np.float64)  # (F, 4)
        self.grab = int(np.argmin(self.hand_mid[:, 2]))   # hands close at min height

    def qpos36(self, f):
        i0 = int(np.floor(f))
        i1 = min(i0 + 1, self.hi - self.lo)
        a = f - i0
        q = np.empty(36)
        q[0:3] = (1 - a) * self.root_p[i0] + a * self.root_p[i1]
        q[3:7] = _nlerp(self.root_q[i0], self.root_q[i1], a)
        jp = (1 - a) * self.jp[i0] + a * self.jp[i1]         # (29,) isaac
        q[7:36] = jp[SP.ISAAC_TO_MUJOCO]
        return q

    def hand(self, f):
        i0 = int(np.floor(f))
        i1 = min(i0 + 1, self.hi - self.lo)
        a = f - i0
        return (1 - a) * self.hand_mid[i0] + a * self.hand_mid[i1]

    def root_yaw(self, f):
        return _yaw_of(self.qpos36(f)[3:7])

    def contact5(self, f):
        row = self.labels[int(np.clip(round(f), 0, self.hi - self.lo))]
        return np.array([row[0], row[1], row[2], row[3], 0.0])


def _bake(smp, frames):
    """One playback (a sequence of fractional clip frames) -> library arrays.

    Returns (qpos (T,36), box_pose (T,7), contact (T,5), attach (T,) bool).
    The box rests at the grab spot while the sampled frame is before the grab,
    and rides the wrist midpoint + root yaw at and after it.
    """
    g = float(smp.grab)
    rest_pos = np.array([smp.hand_mid[smp.grab][0], smp.hand_mid[smp.grab][1],
                         C.BOX_REST_Z])
    yaw_g = smp.root_yaw(g)
    rest_rot = quat.from_angle_axis(np.asarray(yaw_g), UP)   # box axis-aligned to stance
    z_off = C.BOX_REST_Z - smp.hand_mid[smp.grab][2]

    T = len(frames)
    qpos = np.empty((T, 36))
    box_pose = np.empty((T, 7))
    contact = np.empty((T, 5))
    attach = np.zeros(T, bool)
    for k, f in enumerate(frames):
        qpos[k] = smp.qpos36(f)
        contact[k] = smp.contact5(f)
        if f >= g:
            attach[k] = True
            hand = smp.hand(f)
            dyaw = smp.root_yaw(f) - yaw_g
            box_pose[k, 0:3] = [hand[0], hand[1], hand[2] + z_off]
            box_pose[k, 3:7] = quat.mul(
                quat.from_angle_axis(np.asarray(dyaw), UP), rest_rot)
        else:
            box_pose[k, 0:3] = rest_pos
            box_pose[k, 3:7] = rest_rot
    return qpos, box_pose, contact, attach


def build():
    """The two baked playbacks: ('scenebot_pick', ...) at half speed forward and
    ('scenebot_drop', ...) at full speed backward, both at C.FPS (30 Hz).

    Returns a list of (name, qpos, box_pose, contact, attach, phase_code).
    """
    smp = _Sampler()
    F = smp.hi - smp.lo                                    # 120 clip frames
    step_pick = C.SCENEBOT_FPS * C.SCENEBOT_PICK_SPEED / C.FPS    # 5/6
    step_drop = C.SCENEBOT_FPS * C.SCENEBOT_PLACE_SPEED / C.FPS   # 5/3
    fwd = np.arange(0.0, F + 1e-9, step_pick)               # (145,)
    rev = np.arange(float(F), -1e-9, -step_drop)            # (73,) F..0
    out = []
    for name, frames, code in (("scenebot_pick", fwd, Phase.PICK),
                               ("scenebot_drop", rev, Phase.PLACE)):
        qpos, box_pose, contact, attach = _bake(smp, frames)
        out.append((name, qpos, box_pose, contact, attach, code))
    return out
