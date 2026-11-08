"""Port of lmm-g1-box's web/js/phys/runner.js (PhysicalRunner with --ref-mode snap-xyyaw and --hold low) for a robot
that tracks a streamed reference: the deploy node plays 50 Hz frames by index, and the robot is measured by the map
server (SLAM pelvis pose) and by live.sam3box (the box).

The controller runs ahead of the robot by `lookahead` frames. Its 60 Hz poses are resampled to the 50 Hz reference,
and every frame carries the scene of its moment: which box the reference holds and where (ghost_box), and whether
the box controller reaches for a box with empty hands (open). Every REPLAN_TICKS frames the frames the robot has not
tracked yet are dropped and made again from a controller turned and moved onto the robot (seed). The controller
commits 0.2 s at a time: it runs on to the end of the lookahead and goes back to its saved state at the next replan.
While the box controller bends down (HOLDS 'low'), root warping owns the root and the seed waits.

Four changes for a measured robot, all in seed: the robot's pose is compared with the reference frame of the pose's
own time (its age), not with the frame playing now, frames made before the last moves are first carried into the
controller's present coordinates with their corr (the page keeps corr for the same purpose), the seed turns yaw_gain
of the yaw gap (the page: all of it), and with keep_out (on by default) in locomotion no seed puts the simulation
object inside a box's collision zone (kept_out counts them). The controller pushes the simulation object out of the zone at its next step and the
character follows within 0.1 s, so such a seed is undone at once, and seeding that every 0.2 s made the reference
shuffle in place next to the box."""
import math

import numpy as np

from .mathq import hypot, wrap_angle

CONTROL_DT = 0.02
REPLAN_TICKS = 10
SYNC_MOVE = 0.02
SYNC_TURN = 0.05
SYNC_SEARCH = 0.1
GRIP_REF_DZ = 0.4
GRIP_PHYS_DZ = 0.04
GRIP_TIME = 0.1
KEEP_OUT_TOL = 0.05


def lerp_qpos(a, b, t):
    """qpos (36,) or a pose (7,): positions lerped, the quaternion at [3:7] nlerped on the shorter arc."""
    out = a + (b - a) * t
    s = -1.0 if float(a[3:7] @ b[3:7]) < 0 else 1.0
    q = a[3:7] + (s * b[3:7] - a[3:7]) * t
    out[3:7] = q / math.sqrt(float(q @ q))
    return out


def lerp_pose(a, b, t):
    """[x, y, z, qw, qx, qy, qz] poses."""
    if a is None or b is None:
        return b if b is not None else a
    return lerp_qpos(np.asarray(a, float), np.asarray(b, float), t)


def pelvis_yaw(qp):
    """The heading of a pelvis quaternion from its x and z axes together, so a deep forward bend does not flip it."""
    w, x, y, z = qp[3], qp[4], qp[5], qp[6]
    return math.atan2(2 * (x * y + w * z) + 2 * (y * z - w * x), 1 - 2 * (y * y + z * z) + 2 * (x * z + w * y))


def move_corr(c, dyaw, pivot, d):
    """corr [yaw, x, y]: all moves of the controller so far as one turn and shift, a raw pose p at R(yaw) p + (x, y)."""
    co, s, x, y = math.cos(dyaw), math.sin(dyaw), c[1] - pivot[0], c[2] - pivot[1]
    return [c[0] + dyaw, pivot[0] + co * x - s * y + d[0], pivot[1] + s * x + co * y + d[1]]


def to_present(xy, yaw, corr_then, corr_now):
    """A pelvis (xy, yaw) of a frame made with corr_then, in the coordinates of corr_now."""
    a = -corr_then[0]
    c, s = math.cos(a), math.sin(a)
    rx, ry = xy[0] - corr_then[1], xy[1] - corr_then[2]
    raw = (c * rx - s * ry, s * rx + c * ry)
    c, s = math.cos(corr_now[0]), math.sin(corr_now[0])
    return ((c * raw[0] - s * raw[1] + corr_now[1], s * raw[0] + c * raw[1] + corr_now[2]),
            yaw - corr_then[0] + corr_now[0])


HOLDS = {
    'low': lambda c: c.mode == 'box' and c.is_low(),
    'box': lambda c: c.mode == 'box',
    'step': lambda c: c.after_last_step(),
    'none': lambda c: False,
}


class Frame:
    __slots__ = ('qpos', 'held', 'open', 'ghost_box', 'tag', 'corr')

    def __init__(self, qpos, held, open_, ghost_box, tag, corr):
        self.qpos, self.held, self.open, self.ghost_box, self.tag, self.corr = qpos, held, open_, ghost_box, tag, corr


class Capture:
    __slots__ = ('t', 'qpos', 'held', 'open', 'ghost_box', 'tag', 'corr')


class StreamRunner:
    """sim: mm_g1.sim.Sim. Frames are indexed from 0 on the 50 Hz stream clock; frames before `base` are dropped."""

    def __init__(self, sim, lookahead=25, hold='low', kin_dt=1 / 60, keep_s=3.0, yaw_gain=1.0, keep_out=True):
        self.sim = sim
        self.lookahead = lookahead
        self.yaw_gain = yaw_gain
        self.hold = HOLDS[hold]
        self.keep_out = keep_out
        self.kin_dt = kin_dt
        self.keep = int(round(keep_s / CONTROL_DT))
        self.frames = []
        self.base = 0
        self.k = 0
        self.corr = [0.0, 0.0, 0.0]
        self.next_replan = 0
        self.seeds = 0
        self.held_back = 0
        self.kept_out = 0
        self.last_seed = None
        self.kin_curr = self.capture()
        self.kin_prev = self.kin_curr
        self.push(self.kin_curr.qpos, self.kin_curr, self.kin_curr.ghost_box)

    @property
    def produced(self):
        return self.base + len(self.frames)

    def frame_at(self, f):
        return self.frames[max(0, min(f, self.produced - 1) - self.base)]

    def capture(self):
        s, w, c = self.sim, self.sim.world, self.sim.controller
        k = Capture()
        k.t = self.k * self.kin_dt
        k.qpos = np.asarray(s.kin_qpos(), dtype=np.float64)  # (36,)
        hp = s.held_pose if s.held_pose and s.held_pose.get('id') == w.holding_id else None
        k.held = w.holding_id
        k.open = c.mode == 'box' and w.holding_id is None
        k.ghost_box = None if hp is None or w.holding_id is None else np.asarray([*hp['pos'], *hp['rot']], float)
        b = c.backend
        g = None if b.active is None else int(b.db[b.active].idx[b.frame])
        k.tag = dict(s.check_tag(), step=c.box_phase() or c.mode, semantics=c.semantics, g=g, engaged=c.engagedBy)
        k.corr = list(self.corr)
        return k

    def push(self, qpos, kin, ghost_box, corr=None):
        self.frames.append(Frame(qpos, kin.held, kin.open, ghost_box, kin.tag, corr or kin.corr))

    def produce_until(self, j):
        while self.produced <= j:
            t = self.produced * CONTROL_DT
            while self.kin_curr.t < t - 1e-9:
                self.sim.step()
                self.k += 1
                self.kin_prev = self.kin_curr
                self.kin_curr = self.capture()
            p, c = self.kin_prev, self.kin_curr
            a = (t - p.t) / (c.t - p.t) if c.t > p.t else 1.0
            gb = lerp_pose(p.ghost_box, c.ghost_box, a) if p.held == c.held else c.ghost_box
            self.push(lerp_qpos(p.qpos, c.qpos, a), c, gb)

    def commit_state(self):
        return {'F': self.produced - 1, 'sim': self.sim.save(), 'k': self.k, 'kin_curr': self.kin_curr,
                'kin_prev': self.kin_prev, 'corr': list(self.corr)}

    def restore_commit(self, c):
        self.sim.restore(c['sim'])
        self.k, self.kin_curr, self.kin_prev, self.corr = c['k'], c['kin_curr'], c['kin_prev'], list(c['corr'])

    def drop_after(self, F):
        if self.produced > F + 1:
            del self.frames[F + 1 - self.base:]

    def trim(self, F):
        """Drops the frames older than keep_s before F (the seed looks back by the pose's age)."""
        drop = F - self.keep - self.base
        if drop > 0:
            del self.frames[:drop]
            self.base += drop

    def replan(self, F, robot=None, yaw_gain=None):
        """Keeps the frames up to F, the frame the robot plays now, and makes the rest again. robot: None (no seed),
        or (xy, yaw, age_s): the measured pelvis and how old the measurement is. yaw_gain: the seed's, in place of
        the runner's. Returns the first frame made."""
        self.drop_after(F)
        seeded = False
        if robot is not None:
            if self.hold(self.sim.controller):
                self.held_back += 1
            else:
                seeded = self.seed(F, *robot, yaw_gain=yaw_gain)
        self.produce_until(F + REPLAN_TICKS)
        commit = self.commit_state()
        self.sim.speculative = True
        try:
            self.produce_until(F + REPLAN_TICKS + self.lookahead)
        finally:
            self.sim.speculative = False
        self.restore_commit(commit)
        self.next_replan = F + REPLAN_TICKS
        self.trim(F)
        return F + 1, seeded

    def seed(self, F, xy, yaw, age_s=0.0, yaw_gain=None):
        """Turns and moves the controller so that the frames made next start from the robot (snap-xyyaw)."""
        fm = max(self.base, F - int(round(max(0.0, age_s) / CONTROL_DT)))
        f = self.frame_at(fm)
        ref, ref_yaw = to_present(f.qpos[0:2], pelvis_yaw(f.qpos), f.corr, self.corr)
        d = [xy[0] - ref[0], xy[1] - ref[1]]
        dyaw = (self.yaw_gain if yaw_gain is None else yaw_gain) * wrap_angle(yaw - ref_yaw)
        c, fl = self.sim.controller, self.sim.foot_lock
        if self.keep_out and c.mode == 'loco' and self.in_collision_zone(c, dyaw, ref, d):
            self.kept_out += 1
            return False
        if dyaw:
            c.turn(dyaw, ref)
            fl.turn(dyaw, ref)
        c.shift(d)
        fl.shift(d)
        self.corr = move_corr(self.corr, dyaw, ref, d)
        co, sn, r = math.cos(dyaw), math.sin(dyaw), [math.cos(dyaw / 2), 0.0, 0.0, math.sin(dyaw / 2)]

        def pt(p):
            return [ref[0] + co * (p[0] - ref[0]) - sn * (p[1] - ref[1]) + d[0],
                    ref[1] + sn * (p[0] - ref[0]) + co * (p[1] - ref[1]) + d[1], p[2]]

        def rot(w):
            return [r[0] * w[0] - r[3] * w[3], r[0] * w[1] - r[3] * w[2], r[0] * w[2] + r[3] * w[1], r[0] * w[3] + r[3] * w[0]]

        def move(kin):
            out = Capture()
            out.t, out.held, out.open, out.tag = kin.t, kin.held, kin.open, kin.tag
            q = kin.qpos.copy()
            q[0:3] = pt(q[0:3])
            q[3:7] = rot(q[3:7])
            out.qpos = q
            out.ghost_box = None if kin.ghost_box is None else np.asarray([*pt(kin.ghost_box[0:3]), *rot(kin.ghost_box[3:7])])
            out.corr = move_corr(kin.corr, dyaw, ref, d)
            return out

        same = self.kin_prev is self.kin_curr
        self.kin_prev = move(self.kin_prev)
        self.kin_curr = self.kin_prev if same else move(self.kin_curr)
        self.seeds += 1
        self.last_seed = {'frame': fm, 'd': hypot(d[0], d[1]), 'dyaw': dyaw}
        return True

    def in_collision_zone(self, c, dyaw, ref, d):
        """True when the seed would put the simulation object more than KEEP_OUT_TOL inside a box's collision zone."""
        co, sn, sp = math.cos(dyaw), math.sin(dyaw), c.sim['pos']
        moved = [ref[0] + co * (sp[0] - ref[0]) - sn * (sp[1] - ref[1]) + d[0],
                 ref[1] + sn * (sp[0] - ref[0]) + co * (sp[1] - ref[1]) + d[1], sp[2]]
        half = [self.sim.box_size[0] / 2, self.sim.box_size[1] / 2]
        boxes = [{'id': b['id'], 'pos': b['pos'], 'yaw': b['yaw'], 'half': half} for b in self.sim.world.floor_boxes()]
        out = c.collide(moved, boxes)
        return hypot(out[0] - moved[0], out[1] - moved[1]) > KEEP_OUT_TOL

    def pending(self, F, box_id):
        """True when a frame from F on holds box_id: a hold of it is still on its way to the robot."""
        return any(self.frame_at(f).held == box_id for f in range(max(F, self.base), self.produced))


def box_yaw_near(meas_yaw, world_yaw, turn):
    """The copy of a measured box yaw (the box maps onto itself by `turn`) nearest the world's."""
    return meas_yaw + turn * round(wrap_angle(world_yaw - meas_yaw) / turn)


class GripCheck:
    """The grip check of the page (runner.js checkGrip): at the frame the robot plays, the reference's box has risen
    GRIP_REF_DZ while the measured box is within phys_dz of the floor, for need reports in a row (the page reads the
    physics every tick for GRIP_TIME; a camera reports a few times a second)."""

    def __init__(self, half_z, phys_dz=0.06, need=2):
        self.rest = half_z + 0.001
        self.phys_dz = phys_dz
        self.need = need
        self.count = 0
        self.last_stamp = None
        self.held = None

    def update(self, frame, holding_id, meas):
        """meas: the measured box {'pos': (3,), 'yaw', 'stamp'} or None. Returns True when the hold failed."""
        if frame.held != self.held:
            self.held, self.count = frame.held, 0
        if frame.held is None or frame.ghost_box is None or holding_id != frame.held or meas is None:
            return False
        if meas['stamp'] == self.last_stamp:
            return False
        self.last_stamp = meas['stamp']
        failed = frame.ghost_box[2] - self.rest > GRIP_REF_DZ and meas['pos'][2] - self.rest < self.phys_dz
        self.count = self.count + 1 if failed else 0
        if self.count < self.need:
            return False
        self.count = 0
        return True


def sync_box(world_box, meas, box_turn, move_tol=SYNC_MOVE, turn_tol=SYNC_TURN, search=SYNC_SEARCH):
    """The box sync of the page (runner.js syncBoxes) for one box that is not held and has no hold on its way:
    returns None, or (pos, yaw, force_search) to move the world's box to the measured one."""
    yaw = box_yaw_near(meas['yaw'], world_box['yaw'], box_turn)
    move = hypot(meas['pos'][0] - world_box['pos'][0], meas['pos'][1] - world_box['pos'][1])
    turn = abs(wrap_angle(yaw - world_box['yaw']))
    if move > move_tol or turn > turn_tol:
        return [meas['pos'][0], meas['pos'][1]], yaw, move > search or turn > 2 * search
    return None
