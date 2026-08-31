"""The box motion matcher streamed as a SONIC reference motion.

Grows a 50 Hz buffer of matcher output (30 Hz, interpolated) on demand. Each
buffered frame is 43-wide: robot qpos [0:36] + box freejoint pose [36:43].
The properties are the recorded-motion interface SonicPolicy expects (robot
part only).
"""
import numpy as np

from mm_g1 import config as C
from sonic_tracking import params as P

MM_FPS = C.FPS                    # matcher rate (30 Hz)
POLICY_FPS = 1.0 / P.CONTROL_DT   # tracking rate (50 Hz)
WARMUP_TICKS = 60


def nlerp(a, b, t):
    if np.dot(a, b) < 0.0:
        b = -b
    q = (1.0 - t) * a + t * b
    return q / (np.linalg.norm(q) + 1e-12)


class MMMotion:
    def __init__(self, matcher, commander):
        self.matcher = matcher
        self.commander = commander            # (matcher) -> (vel, face), per tick
        self.pre_tick = None                  # hook (the 'anchor' ref mode)
        self._mm_t0 = None
        self._mm_t1 = None
        self._ticks = 0.0
        for _ in range(WARMUP_TICKS):
            q = self._full_qpos(matcher.step(np.zeros(3), np.zeros(3)))
        self._mm_t0 = self._mm_t1 = q
        self._frames = [q.copy()]
        self._meta = [self._meta_now()]
        self._rebuild()

    def _full_qpos(self, robot_q):
        return np.concatenate([robot_q, self.matcher.box_qpos()])  # (43,)

    def _meta_now(self):
        m = self.matcher
        return (int(m.lib['clip_id'][m.cur]), int(m.lib['frame_in_clip'][m.cur]),
                m.state, bool(m.box_held))

    def meta_at(self, f):
        """Matcher clip/frame/state/held as recorded at buffer frame `f` --
        the matcher itself has since run ahead of it."""
        return self._meta[min(f, len(self._meta) - 1)]

    def _step_matcher(self):
        if self.pre_tick is not None:
            self.pre_tick()
        vel, face = self.commander(self.matcher)
        q = self._full_qpos(self.matcher.step(vel, face))
        self._mm_t0, self._mm_t1 = self._mm_t1, q
        self._ticks += 1.0

    def ensure(self, upto):
        grew = False
        while len(self._frames) < upto:
            t = len(self._frames) / POLICY_FPS
            while self._ticks / MM_FPS < t:
                self._step_matcher()
            t0 = (self._ticks - 1) / MM_FPS
            mix = float(np.clip((t - t0) * MM_FPS, 0.0, 1.0))
            q = (1.0 - mix) * self._mm_t0 + mix * self._mm_t1
            q[3:7] = nlerp(self._mm_t0[3:7], self._mm_t1[3:7], mix)
            q[39:43] = nlerp(self._mm_t0[39:43], self._mm_t1[39:43], mix)
            self._frames.append(q)
            self._meta.append(self._meta_now())
            grew = True
        if grew:
            self._rebuild()

    # Replan support: time_mark/time_restore let a roll-forward past the
    # committed period be undone.
    def truncate(self, n):
        del self._frames[n:]
        del self._meta[n:]
        self._ticks = len(self._frames) / POLICY_FPS * MM_FPS
        q = self._frames[-1].copy()
        self._mm_t0 = self._mm_t1 = q
        self._rebuild()

    def time_mark(self):
        return (self._ticks, self._mm_t0.copy(), self._mm_t1.copy())

    def time_restore(self, mark):
        self._ticks, self._mm_t0, self._mm_t1 = \
            mark[0], mark[1].copy(), mark[2].copy()

    def _rebuild(self):
        self._qpos = np.asarray(self._frames)
        angles = self._qpos[:, 7:36][:, P.MUJOCO_TO_ISAACLAB]
        speeds = np.zeros_like(angles)
        if len(angles) > 1:
            speeds[1:] = (angles[1:] - angles[:-1]) * POLICY_FPS
        self._joint_pos, self._joint_vel = angles, speeds

    @property
    def timesteps(self):
        return self._qpos.shape[0]

    @property
    def qpos(self):
        return self._qpos

    @property
    def joint_pos(self):
        return self._joint_pos

    @property
    def joint_vel(self):
        return self._joint_vel

    @property
    def body_pos(self):
        return self._qpos[:, None, 0:3]

    @property
    def body_quat(self):
        return self._qpos[:, None, 3:7]
