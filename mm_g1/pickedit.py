"""Port of the part of lmm-g1-box's web/js/edit/pickedit.js that pickmotion.js reaches: the pick editor's motion
model. The source is one pick: the robot stops in front of a box at rest, bends down, grasps it (attach) and rises
with it. Every frame is solved by whole-body IK (bodyik.py) from the source frame, with these targets:

  box      before the attach the box rests at its edited place. From the attach on it is the source box of the
           frame, moved by the share of the edit that is left: 1 minus the share of the lift the source box has done.
  palms    the source palm in the frame of the source box, put into a box that moves from the source's place to the
           edited one: sideways and in yaw by a share that rises from 0 at blendStart to 1 `lead` frames before the
           attach, in height by the share of the way down the source palms have gone. From the attach on that box is
           the held one.
  feet     the source feet of the frame.
  posture and centre of mass   those of the reference pose of the frame. Without a stand it is the source frame.
           With a stand the robot bends less, so the reference is the source at another time: before the attach the
           frame where the palms are as far down as the frame's share of the shorter way, after it the frame where
           the box is as high as the edited box. Their difference comes in over the last SETTLE frames before the
           attach.

Left out of the JS, because the pick motion's edits never turn them on (context() raises if one is asked for):

  pose keys    DEFAULT_KEY's hands/feet/joints/links/lean/free, keyWeight, keyState, keyTargets and the second
               solve of a keyed frame. The pick motion builds its edits with keys: [].
  symmetry     symmetricKey, mirrored()'s blend of a foot with the other foot, towardMirrorQpos/towardMirrorPose in
               boxPose and frameTargets, and palmInBox's mirror branch. editOf gives symmetry.on = False, so the
               shares c.sym and sh.sym are 0 everywhere.
  feet keep / stance   context()'s c.keep and c.stance and the two branches of footTarget that use them, with
               Plane.mirrorPoint for the kept pose. editOf gives feet.keep = False and a zero stance.
  timing       timeMap, isRetimed's effect and the resampling at the end of solve(). isRetimed itself is kept, since
               isNoEdit calls it. editOf gives scale [1, 1, 1, 1] and no pause.
  check()      the kinematic checks of tools/verify_pick.py, reversed() and joined(), centredLeft() and frameOf():
               nothing in pickmotion.js calls them.

The class holds a BodyModel, so one instance is not safe to share between threads.
"""
import math

import numpy as np

from .bodyik import DEFAULT_WEIGHTS, posture_weights
from .mathq import qfrom_yaw, qinv_mul, qmul, qmul_vec, qslerp_shortest_approx
from .mirror import Plane, mat_quat, quat_mat
from .runner import lerp_qpos, pelvis_yaw

STAND_MIN = 0.005
SETTLE = 30

# With the grip kept, the palm's orientation weighs this many times more in the IK of the held frames.
GRIP_ROT = 10

DEFAULT_EDIT = {
    'forward': 0, 'left': 0, 'up': 0, 'yaw': 0,
    'alignBox': True,
    'grasp': [[0, 0], [0, 0]],
    'blendStart': None, 'lead': 10,
    'standSize': None,
    'box': None,
    'fitBox': True,
    'grip': {'keep': False},
    'symmetry': {'on': False, 'feet': True, 'fromStart': True},
    'feet': {'keep': False, 'from': 0, 'to': None, 'flat': True, 'lift': 0, 'stance': [[0, 0, 0], [0, 0, 0]]},
    'keys': [],
    'timing': {'scale': [1, 1, 1, 1], 'pauseBefore': 0, 'pauseAfter': 0},
}


def _any(v):
    return any(x != 0 for x in v)


def edit_of(e=None):
    e = e or {}
    sym = e.get('symmetry') or {}
    feet = e.get('feet') or {}
    timing = e.get('timing') or {}
    out = dict(DEFAULT_EDIT, **e)
    out['grasp'] = [list(g) for g in e.get('grasp', DEFAULT_EDIT['grasp'])]
    out['standSize'] = list(e['standSize']) if e.get('standSize') else None
    out['box'] = list(e['box']) if e.get('box') else None
    out['grip'] = dict(DEFAULT_EDIT['grip'], **(e.get('grip') or {}))
    # An edit saved before the symmetry was on or off has its share as `amount`.
    out['symmetry'] = {
        'on': sym['on'] if 'on' in sym else sym.get('amount', 0) >= 0.5,
        'feet': sym.get('feet', True),
        'fromStart': sym.get('fromStart', 'amount' not in sym),
    }
    out['feet'] = dict(DEFAULT_EDIT['feet'], **feet)
    out['feet']['stance'] = [list(f) for f in feet.get('stance', DEFAULT_EDIT['feet']['stance'])]
    out['keys'] = list(e.get('keys') or [])
    out['timing'] = dict(DEFAULT_EDIT['timing'], **timing)
    out['timing']['scale'] = list(timing.get('scale', DEFAULT_EDIT['timing']['scale']))
    return out


def is_retimed(e):
    t = e['timing']
    return any(s != 1 for s in t['scale']) or t['pauseBefore'] > 0 or t['pauseAfter'] > 0


def is_no_edit(e):
    return (not e['forward'] and not e['left'] and not e['up'] and not e['yaw'] and not e['alignBox']
            and all(not g[0] and not g[1] for g in e['grasp']) and not e['box'] and not e['symmetry']['on']
            and not e['feet']['keep'] and not any(_any(f) for f in e['feet']['stance'])
            and not e['keys'] and not is_retimed(e))


def smoothstep(x):
    t = min(1.0, max(0.0, x))
    return t * t * (3 - 2 * t)


def mat_mul(A, B):
    C = [0.0] * 9
    for i in range(3):
        for j in range(3):
            C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j]
    return C


def mat_T(A):
    return [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]]


def mat_vec(A, v):
    return [A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
            A[3] * v[0] + A[4] * v[1] + A[5] * v[2],
            A[6] * v[0] + A[7] * v[1] + A[8] * v[2]]


def dist3(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])


def pose_yaw(p):
    """runner.js poseYaw: the yaw of a box pose, from whichever of its local x and y axes lies flatter."""
    w, x, y, z = p[3], p[4], p[5], p[6]
    ax = [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)]
    if abs(ax[2]) < 0.7:
        return math.atan2(ax[1], ax[0])
    ay = [2 * (x * y - w * z), 1 - 2 * (x * x + z * z)]
    return math.atan2(ay[1], ay[0]) - math.pi / 2


def _sign(v, zero):
    return 1.0 if v > 0 else -1.0 if v < 0 else zero


class PickEdit:
    def __init__(self, model, source):
        self.model = model
        self.source = source
        self.T = len(source['qpos'])
        self.fps = source['fps']
        self.attach = source['attach']
        self.lifted = source['lifted']
        self.srcHalf = [s / 2 for s in source['box_size_data']]
        self.half = self.srcHalf
        self.qpos = [np.array(r, dtype=np.float64) for r in source['qpos']]  # (T, 36)
        self.box = [list(r) for r in source['box']]                          # (T, 7)
        self.heading = pelvis_yaw(self.qpos[self.attach])
        self.restYaw = pose_yaw(self.box[self.attach - 1])
        self.alignTurn = math.atan2(math.sin(self.heading - self.restYaw), math.cos(self.heading - self.restYaw))
        z0, z1 = self.box[self.attach - 1][2], self.box[self.T - 1][2]
        share = 0.0
        self.lift = []  # (T,) the share of the lift the source box has done, never falling
        for t, b in enumerate(self.box):
            if t < self.attach:
                self.lift.append(0.0)
            else:
                share = max(share, min(1.0, (b[2] - z0) / (z1 - z0)))
                self.lift.append(share)
        self.lift[self.T - 1] = 1.0
        self.liftHeight = z1 - z0
        stop, A = source['stop'], self.attach
        zp = []  # (T,) the mean palm height of the source frame
        for qp in self.qpos:
            model.set_robot(qp)
            zp.append((model.palm(0)['p'][2] + model.palm(1)['p'][2]) / 2)
        self.reachDepth = zp[stop] - zp[A]
        down = 0.0
        self.reach = []  # (T,) the share of the way down the source palms have gone, never falling
        for t, z in enumerate(zp):
            if t <= stop:
                self.reach.append(0.0)
            elif t >= A:
                self.reach.append(1.0)
            else:
                down = max(down, min(1.0, (zp[stop] - z) / self.reachDepth))
                self.reach.append(down)
        self.targets = []
        for t, qp in enumerate(self.qpos):
            tg = model.targets(qp)
            b = self.box[t]
            RbT = mat_T(quat_mat(b[3:7]))
            tg['palmInBox'] = [{'p': mat_vec(RbT, [h['p'][0] - b[0], h['p'][1] - b[1], h['p'][2] - b[2]]),
                                'q': mat_quat(mat_mul(RbT, h['R']))} for h in tg['palm']]
            self.targets.append(tg)
        rest, pelvis = self.box[A - 1], self.qpos[A]
        ax = qmul_vec(rest[3:7], [1.0, 0.0, 0.0])
        # which side of the box along its x axis the robot stands on: -1 or 1
        self.near = _sign((pelvis[0] - rest[0]) * ax[0] + (pelvis[1] - rest[1]) * ax[1], -1.0)
        self.palmSide = [_sign(h['p'][1], 1.0) for h in self.targets[A]['palmInBox']]

    def context(self, edit):
        """What an edit fixes for every frame: the box size and the mirror plane."""
        if edit['symmetry']['on'] or edit['feet']['keep'] or any(_any(f) for f in edit['feet']['stance']) \
                or edit['keys'] or is_retimed(edit):
            raise NotImplementedError('this port solves the pick motion edits only: no symmetry, kept feet, '
                                      'foot stance, pose keys or retiming')
        fit = bool(edit['box'] and edit['fitBox'])
        half = [s / 2 for s in edit['box']] if fit else self.srcHalf
        c = {'edit': edit, 'half': half, 'fit': fit,
             'dHalf': [h - self.srcHalf[i] for i, h in enumerate(half)],
             'ratio': [h / self.srcHalf[i] for i, h in enumerate(half)]}
        c['sizeShift'] = [-self.near * c['dHalf'][0], 0.0, c['dHalf'][2]]
        rest = self.rest_box(c)
        c['plane'] = Plane(rest, pose_yaw(rest))
        c['sym'] = 0
        c['symStart'] = bool(edit['symmetry']['fromStart'])
        c['footSym'] = False
        c['keys'] = []
        h = self.heading
        c['fwd'] = [math.cos(h), math.sin(h)]
        c['lft'] = [-math.sin(h), math.cos(h)]
        c['keep'] = None
        c['stance'] = None
        return c

    def mirrored(self, c, t, s, g):
        return self.targets[t]['foot'][s]

    def feet_left(self, c):
        """How far to the left of the mirror plane the middle of the feet is at the attach (m)."""
        f = self.targets[self.attach]['foot']
        return c['plane'].local([(f[0]['p'][0] + f[1]['p'][0]) / 2, (f[0]['p'][1] + f[1]['p'][1]) / 2])[1]

    def shares(self, edit, t):
        """The shares of the edit that the palm targets of frame t have."""
        if t >= self.attach:
            u = 1 - self.lift[t]
            return {'side': u, 'up': u, 'grasp': 1, 'size': 1, 'sym': 1}
        a = edit['blendStart'] if edit['blendStart'] is not None else self.source['stop']
        b = max(a + 1, self.attach - edit['lead'])
        side = smoothstep((t - a) / (b - a))
        return {'side': side, 'up': self.reach[t], 'grasp': side, 'size': side,
                'sym': 1 if edit['symmetry']['fromStart'] else side}

    def shift(self, edit):
        c, s = math.cos(self.heading), math.sin(self.heading)
        return [c * edit['forward'] - s * edit['left'], s * edit['forward'] + c * edit['left'], edit['up']]

    def moved(self, c, t, side, up, size):
        """The source box of frame t with the share `size` of the change of its size, then moved by the shares."""
        b = self.box[t]
        d = self.shift(c['edit'])
        bq = b[3:7]
        o = qmul_vec(bq, [size * c['sizeShift'][0], 0.0, size * c['sizeShift'][2]]) if size else [0.0, 0.0, 0.0]
        return [b[0] + o[0] + side * d[0], b[1] + o[1] + side * d[1], b[2] + o[2] + up * d[2]] \
            + qmul(qfrom_yaw(side * c['edit']['yaw'] + self.align_of(c['edit'])), bq)

    def align_of(self, edit):
        return self.alignTurn if edit['alignBox'] else 0

    def rest_box(self, c):
        return self.moved(c, self.attach - 1, 1, 1, 1)

    def box_pose(self, c, t):
        if t < self.attach:
            return self.moved(c, t, 1, 1, 1)
        u = 1 - self.lift[t]
        return self.moved(c, t, u, u, 1)

    def stand(self, c):
        """The stand under the box at rest, or None when the box rests on the floor."""
        edit = c['edit']
        if edit['up'] < STAND_MIN:
            return None
        b = self.rest_box(c)
        size = edit['standSize'] if edit['standSize'] is not None else [2 * c['half'][0], 2 * c['half'][1]]
        return {'pos': [b[0], b[1], edit['up'] / 2], 'yaw': pose_yaw(b), 'size': [size[0], size[1], edit['up']]}

    def frame_at(self, series, share, start, to):
        """The first frame (with a fraction) from `start` on where the never-falling series reaches the share."""
        for t in range(start, to):
            if series[t + 1] >= share:
                a, b = series[t], series[t + 1]
                return t + min(1.0, max(0.0, (share - a) / (b - a))) if b > a else t + (0 if a >= share else 1)
        return to

    def source_at(self, x):
        i = min(self.T - 1, max(0, int(math.floor(x))))
        j = min(self.T - 1, i + 1)
        return np.array(self.qpos[i]) if x - i < 1e-9 else lerp_qpos(self.qpos[i], self.qpos[j], x - i)

    def warp_reach(self, edit, t):
        A, stop = self.attach, self.source['stop']
        r = self.reach[t]
        shorter = max(0.0, 1 - edit['up'] / self.reachDepth)
        return min(A, max(stop, t + self.frame_at(self.reach, shorter * r, stop, A)
                          - self.frame_at(self.reach, r, stop, A)))

    def warp_lift(self, edit, t):
        A = self.attach
        l = self.lift[t]
        higher = min(1.0, edit['up'] / self.liftHeight + l * (1 - edit['up'] / self.liftHeight))
        return min(self.T - 1, max(A, t + self.frame_at(self.lift, higher, A, self.T - 1)
                                   - self.frame_at(self.lift, l, A, self.T - 1)))

    def reference(self, edit, t):
        """The reference pose of frame t: (36,) float64, or None for the source frame itself."""
        if edit['up'] < STAND_MIN or t <= self.source['stop']:
            return None
        A = self.attach
        if t >= A:
            return self.source_at(self.warp_lift(edit, t))
        q = self.source_at(self.warp_reach(edit, t))
        s = smoothstep((t - (A - SETTLE)) / SETTLE)
        if s <= 0:
            return q
        a = self.source_at(self.warp_reach(edit, A))
        b = self.source_at(self.warp_lift(edit, A))
        for i in range(3):
            q[i] += s * (b[i] - a[i])
        for i in range(7, 36):
            q[i] += s * (b[i] - a[i])
        turn = qinv_mul(list(a[3:7]), list(b[3:7]))
        q[3:7] = qmul(list(q[3:7]), qslerp_shortest_approx([1.0, 0.0, 0.0, 0.0], turn, s))
        return q

    def palm_in_box(self, c, t, s, sh):
        """The source palm of hand s at frame t in the frame of its box, for the box size of the context."""
        def sized(k):
            h = self.targets[t]['palmInBox'][k]['p']
            z = sh['size']
            if not c['fit'] or not z:
                return list(h)
            return [h[0] * (1 + z * (c['ratio'][0] - 1)), h[1] + self.palmSide[k] * z * c['dHalf'][1],
                    h[2] * (1 + z * (c['ratio'][2] - 1))]
        return {'p': sized(s), 'q': self.targets[t]['palmInBox'][s]['q']}

    def foot_target(self, c, t, s):
        F = self.mirrored(c, t, s, 1)
        return {'p': F['p'], 'R': F['R'], 'd': [0, 0], 'e': [0, 0], 'moved': False}

    def frame_targets(self, c, t):
        edit = c['edit']
        src = self.targets[t]
        sh = self.shares(edit, t)
        held = t >= self.attach
        box = self.box_pose(c, t)
        foot = [self.foot_target(c, t, 0), self.foot_target(c, t, 1)]
        b = box if held else self.moved(c, t, sh['side'], sh['up'], sh['size'] if c['fit'] else 0)
        bq = b[3:7]
        gripped = held and edit['grip']['keep']
        palm = []
        for s in (0, 1):
            h = self.palm_in_box(c, self.attach if gripped else t, s, sh)
            g = edit['grasp'][s]
            p = qmul_vec(bq, [h['p'][0] + sh['grasp'] * g[0], h['p'][1], h['p'][2] + sh['grasp'] * g[1]])
            palm.append({'p': [b[0] + p[0], b[1] + p[1], b[2] + p[2]], 'R': quat_mat(qmul(bq, h['q']))})
        moved = (sh['side'] > 0 and (edit['forward'] or edit['left'] or edit['yaw'])) \
            or (sh['up'] > 0 and edit['up']) or self.align_of(edit) != 0
        shifted = sh['grasp'] > 0 and any(g[0] or g[1] for g in edit['grasp'])
        sized = c['fit'] and sh['size'] > 0 and _any(c['dHalf'])
        ref = self.reference(edit, t)
        com, q = src['com'], src['q'] if ref is None else ref
        if ref is not None:
            self.model.set_robot(ref)
            com = self.model.com()
        return {'foot': foot, 'palm': palm, 'palmRotW': GRIP_ROT if gripped else 1, 'com': com, 'q': q, 'box': box,
                'changed': bool(moved or shifted or sized or ref is not None or gripped)}

    def solve(self, edit, weights=None, iters=12):
        """The edited pick: qpos, box, hold, miss, src per frame, with the clip's numbers."""
        weights = weights or DEFAULT_WEIGHTS
        posture = posture_weights(weights)
        m, c, A = self.model, self.context(edit), self.attach
        S = {'qpos': [], 'box': [], 'miss': []}
        for t in range(self.T):
            T = self.frame_targets(c, t)
            q = np.array(T['q'], dtype=np.float64)  # (36,)
            if T['changed']:
                m.solve(q, T, weights, posture, iters)
            m.set_robot(q)
            com = m.com()
            S['miss'].append({
                'palm': [dist3(m.palm(s)['p'], T['palm'][s]['p']) for s in (0, 1)],
                'foot': [dist3(m.foot(s)['p'], T['foot'][s]['p']) for s in (0, 1)],
                'com': math.hypot(com[0] - T['com'][0], com[1] - T['com'][1]),
                'pelvis': 0, 'link': 0, 'joint': 0,
            })
            S['qpos'].append(q)
            S['box'].append(T['box'])
        marks = {'stop': self.source['stop'], 'attach': A, 'lifted': self.lifted}
        # Where the palms hold the box at the attach: their middle above the box's centre, along the box's up axis.
        m.set_robot(S['qpos'][A])
        bA = S['box'][A]
        upA = qmul_vec(bA[3:7], [0.0, 0.0, 1.0])
        grip = 0.0
        for s in (0, 1):
            p = m.palm(s)['p']
            grip += (upA[0] * (p[0] - bA[0]) + upA[1] * (p[1] - bA[1]) + upA[2] * (p[2] - bA[2])) / 2
        return {
            'fps': self.fps, 'stop': marks['stop'], 'attach': A, 'lifted': self.lifted, 'solved': S, 'marks': marks,
            'stand': self.stand(c), 'boxSize': [2 * h for h in c['half']],
            'physBox': list(edit['box']) if edit['box'] else list(self.source['box_size_physics']),
            'plane': c['plane'], 'palm0': [self.palm_in_box(c, A, s, {'size': 1, 'sym': 1})['p'] for s in (0, 1)],
            'grip': grip, 'feetLeft': self.feet_left(c),
            'qpos': S['qpos'], 'box': S['box'], 'miss': S['miss'],
            'hold': [t >= A for t in range(self.T)], 'src': [float(t) for t in range(self.T)],
        }
