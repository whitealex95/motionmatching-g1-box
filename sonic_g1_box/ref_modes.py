"""Keeping the kinematic motion-matching root consistent with the robot.

SONIC never observes root XY, so the robot drifts from the reference.
--ref-mode picks one of five ways to close that loop: `anchor` (capped
per-tick correction, never truncates), or the replan family `anchor-replan` /
`snap-xy` / `snap-xyyaw` / `snap-all` (truncate + redraw, root seeded from
the robot). Both worlds share one frame (heading alignment pinned to
identity).
"""
import numpy as np

from mm_g1.features import yaw_quat

MODES = ('anchor', 'anchor-replan', 'snap-xy', 'snap-xyyaw', 'snap-all')
REPLAN_MODES = MODES[1:]


def yaw_of(q):
    q = np.asarray(q, float)
    return float(np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))


# Arrays this size or larger are the read-only motion database; share them
# instead of copying a 40k-row table at every snapshot.
_SHARED_ARRAY_MIN = 4096


def snapshot(ctrl):
    """Copy of the matcher's mutable state, so a rollout can be undone."""
    out = {}
    for k, v in ctrl.__dict__.items():
        if isinstance(v, np.ndarray) and v.size < _SHARED_ARRAY_MIN:
            out[k] = v.copy()
        elif isinstance(v, set):
            out[k] = set(v)
        else:
            out[k] = v
    return out


def restore(ctrl, snap):
    ctrl.__dict__.update(snap)


class RootSeeder:
    def __init__(self, mode, matcher, data, stream,
                 gain=0.738, q_at=None, dq_at=None):
        if mode not in REPLAN_MODES:
            raise ValueError(f'RootSeeder handles {REPLAN_MODES}, not {mode!r}')
        self.mode = mode
        self.matcher, self.data, self.stream = matcher, data, stream
        self.gain = float(gain)
        self.q_at, self.dq_at = q_at, dq_at
        self.err = []

    def seed(self, f):
        """Place the matcher so the plan continues from the robot. Call AFTER
        stream.truncate(f + 1), with `f` the frame the robot is tracking."""
        m, d = self.matcher, self.data
        robot_xy = d.qpos[0:2].copy()
        plan_xy = self.stream.qpos[min(f, self.stream.timesteps - 1)][0:2]
        self.err.append(float(np.linalg.norm(robot_xy - plan_xy)))

        if self.mode == 'anchor-replan':
            m.rootPos[0:2] = plan_xy + self.gain * (robot_xy - plan_xy)
            return

        m.rootPos[0:2] = robot_xy
        if self.mode in ('snap-xyyaw', 'snap-all'):
            m.rootYaw = yaw_of(d.qpos[3:7])
            m.rootRot = yaw_quat(m.rootYaw)

        if self.mode == 'snap-all':
            fr = m.animFrame
            m.offDof = d.qpos[self.q_at] - m.dof[fr]
            m.offDofVel = d.qvel[self.dq_at] - m.dofVel[fr]
            m.offPP = m.offPP.copy()
            m.offPP[2] = float(d.qpos[2]) - float(m.plpDB[fr][2])
            m.offPPVel = m.offPPVel.copy()
            m.offPPVel[2] = float(d.qvel[2]) - float(m.plvDB[fr][2])

    def summary_lines(self):
        if not self.err:
            return []
        e = np.array(self.err)
        return [f'[root] mode {self.mode}: pre-seed error mean {e.mean():.3f} m, '
                f'p95 {np.percentile(e, 95):.3f} m, max {e.max():.3f} m '
                f'({len(e)} projections; '
                + (f'moves {self.gain:g} of the way'
                   if self.mode == 'anchor-replan'
                   else 'zero immediately after each seed') + ')']


class ContinuousAnchor:
    """The ``anchor`` mode: a correction every matcher tick, no truncation.

    The drift is measured at the frame the robot is tracking, which was
    emitted ~lookahead ago, so the total already applied since that frame is
    subtracted from the measurement before the gain (a Smith predictor).
    """

    CAP = 0.03                            # m per 30 Hz tick; runaway guard

    def __init__(self, matcher, data, stream, policy, gain=0.20):
        self.matcher, self.data = matcher, data
        self.stream, self.policy = stream, policy
        self.gain = float(gain)
        self.applied_total = np.zeros(2)
        self.applied_at_frame = []
        self.err = []

    def record_applied(self):
        while len(self.applied_at_frame) < self.stream.timesteps:
            self.applied_at_frame.append(self.applied_total.copy())

    def before_matcher_tick(self):
        if self.gain <= 0.0:
            return
        self.record_applied()
        f = min(self.policy.current_frame, self.stream.timesteps - 1)
        drift = self.data.qpos[0:2] - self.stream.qpos[f][0:2]
        self.err.append(float(np.linalg.norm(drift)))
        corr = self.gain * (drift -
                            (self.applied_total - self.applied_at_frame[f]))
        n = float(np.linalg.norm(corr))
        if n > self.CAP:
            corr *= self.CAP / n
        self.matcher.rootPos[0:2] += corr
        self.applied_total += corr

    def summary_lines(self):
        if not self.err:
            return []
        e = np.array(self.err)
        return [f'[root] mode anchor (continuous, gain {self.gain:g}): '
                f'drift mean {e.mean():.3f} m, p95 {np.percentile(e, 95):.3f} m, '
                f'max {e.max():.3f} m ({len(e)} corrections)']
