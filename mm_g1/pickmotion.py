"""Port of lmm-g1-box's web/js/mm/pickmotion.js: the pick motion that plays. The SceneBot pick and place of the data
variant, solved again by the pick editor's whole-body IK (pickedit.py) for the box the robot goes for. The Matching
Databases, the costs and the search stay as they are. Only the rows of the Animation Database that play these clips
change: the pelvis, the joints, the box and their velocities. The root, the contacts and the hold state stay the
data's.

The put down uses the manual edit of the pick up, the default, or its own box at rest and palms (manualPlace). The box
is one for both. While a bend plays, Sim.sync_pick_motion gives both clips the edit of that bend, so a cancel that goes
on in the other clip keeps the palms where they are.

The source of an edit is the clip as the database has it, at 60 fps, from the frame where the walk ends. A place is
solved as a pick: its rows played backward are a pick, and the solved frames go back in the place's order.

Threads: an instance has no global state and may run off the control loop, but it is not itself thread safe. One
lock has to cover
  * every call that writes backend.pose, that is use(), sync() and restore(), against the control loop reading it,
  * solved(), prepare(), shown(), set_manual() and palm_range(), which share the one BodyModel and the caches (cache, sources, roots),
  * the caller's own reads of info / offsets() / lift().
A worker that solves ahead can hold the lock for prepare() only, and then take it again for the short sync().
"""
import json
import math
import time

import numpy as np

from .bodyik import BodyModel
from .mathq import (qabs, qfrom_yaw, qinv_mul, qmul, qmul_inv, qnorm, qslerp_shortest_approx,
                    qto_scaled_angle_axis, qyaw)
from .pickedit import PickEdit, edit_of, is_no_edit

MANUAL = {'forward': 0, 'left': 0, 'up': 0, 'yaw': 0, 'grasp': [[0, 0], [0, 0]], 'gripKeep': False, 'box': None}
PICK_MOTION = {'autoHeight': True, 'autoSize': False}
# The limits of the box at rest, in the units of the page's sliders and of its address: cm, cm, cm, degrees.
REST_LIMITS = {'forward': (-15, 25), 'left': (-20, 20), 'up': (0, 70), 'yaw': (-45, 45)}
REST_KEYS = ('forward', 'left', 'up', 'yaw')
DEG = 180 / math.pi
# A support lower than this is the floor. Support heights are rounded to this for the cache.
SUPPORT_MIN = 0.005
SIZE_TOL = 1e-4
FACE_MARGIN = 0.02

CLIP = {'pick': 'SceneBotPick', 'place': 'SceneBotPlace'}


def scene_bot_clips(cfg, box_db):
    """The SceneBot pick and place rows of the Animation Database that a range of the box Matching Database plays."""
    names, rows = cfg['clips']['names'], cfg['clips']['rows']
    idx, ranges = box_db.idx, np.asarray(box_db.ranges).reshape(-1, 2)
    out = []
    for row, (n, mirror, start, stop, attach, release) in enumerate(rows):
        kind = 'pick' if names[n] == CLIP['pick'] else 'place' if names[n] == CLIP['place'] else None
        if not kind:
            continue
        used = False
        for a, b in ranges:
            if used:
                break
            used = idx[a] < stop and idx[b - 1] >= start
        if used:
            out.append({'kind': kind, 'row': row, 'mirror': mirror, 'start': start, 'stop': stop,
                        'attach': attach, 'release': release})
    return out


def own_manual(m):
    """The part of a manual edit that the put down can have of its own."""
    return {'forward': m['forward'], 'left': m['left'], 'up': m['up'], 'yaw': m['yaw'],
            'grasp': [list(g) for g in m['grasp']], 'gripKeep': m['gripKeep']}


def _numbers(text, counts, what):
    try:
        v = [float(t) for t in str(text).split(',')]
    except ValueError:
        v = None
    if v is None or len(v) not in counts or not all(math.isfinite(x) for x in v):
        raise ValueError(f'bad {what} "{text}"')
    return v


def _round1(v):
    """+v.toFixed(1) of JS."""
    return float('%.1f' % v) if v >= 0 else -float('%.1f' % -v)


def central_diff(x, fps):
    n = len(x)
    v = np.zeros(n)  # (T,)
    for t in range(1, n - 1):
        v[t] = 0.5 * (x[t + 1] - x[t - 1]) * fps
    if n >= 4:
        v[0] = v[1] - (v[3] - v[2])
        v[n - 1] = v[n - 2] + (v[n - 2] - v[n - 3])
    return v


def angular_velocity(Q, fps):
    """lmm/database.py angular_velocity: per frame [x, y, z] (rad/s) of a series of quaternions."""
    n = len(Q)
    w = [[0.0, 0.0, 0.0] for _ in range(n)]  # (T, 3)

    def half(a, b):
        return [0.5 * v * fps for v in qto_scaled_angle_axis(qabs(qmul_inv(a, b)))]

    for t in range(1, n - 1):
        a, b = half(Q[t + 1], Q[t]), half(Q[t], Q[t - 1])
        w[t] = [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    if n >= 4:
        for i in range(3):
            w[0][i] = w[1][i] - (w[3][i] - w[2][i])
            w[n - 1][i] = w[n - 2][i] + (w[n - 2][i] - w[n - 3][i])
    return w


def unroll(Q):
    out = [list(Q[0])]
    for t in range(1, len(Q)):
        p, c = out[t - 1], Q[t]
        out.append([-v for v in c] if p[0] * c[0] + p[1] * c[1] + p[2] * c[2] + p[3] * c[3] < 0 else list(c))
    return out


def upsample(data, solved, k):
    """The 60 fps frames of a clip solved at 30 fps: a captured frame is the solved one, a frame between two
    captured ones is the data's frame moved by the mean of the edit of its neighbours. data: (T, n) at 60 fps,
    solved: ((T + 1) / 2, n). The quaternion at k..k+4 is composed."""
    T = len(data)
    out = [None] * T

    def turn(i):
        return qmul_inv(list(solved[i][k:k + 4]), list(data[2 * i][k:k + 4]))

    for t in range(T):
        if t % 2 == 0:
            out[t] = np.array(solved[t // 2], dtype=np.float64)
            continue
        a = (t - 1) // 2
        b = a + 1
        o = np.array(data[t], dtype=np.float64)
        for i in range(len(o)):
            if k <= i < k + 4:
                continue
            o[i] += 0.5 * ((solved[a][i] - data[2 * a][i]) + (solved[b][i] - data[2 * b][i]))
        r = qmul(qslerp_shortest_approx(turn(a), turn(b), 0.5), list(data[t][k:k + 4]))
        o[k:k + 4] = r
        out[t] = o
    return out


def _to_fixed3(v):
    """Number(v.toFixed(3)) of JS: half away from zero on the decimal string."""
    return float('%.3f' % v) if v >= 0 else -float('%.3f' % -v)


class PickMotion:
    def __init__(self, cfg, backend, scene_xml_path, sonic_json_path):
        """scene_xml_path / sonic_json_path: data/physics of the page. Either None gives no IK model, where every
        clip plays as the data has it, as the page does when loadIK fails."""
        self.cfg = cfg
        self.backend = backend
        self.model = None
        if scene_xml_path and sonic_json_path:
            with open(scene_xml_path) as f:
                xml = f.read()
            with open(sonic_json_path) as f:
                meta = json.load(f)
            self.model = BodyModel(xml, meta)
        self.settings = dict(PICK_MOTION)
        self.manual = dict(MANUAL, grasp=[list(g) for g in MANUAL['grasp']])
        # None: the put down uses the manual edit of the pick up.
        self.manualPlace = None
        self.dataBox = list(cfg['box_size'])
        self.clips = scene_bot_clips(cfg, backend.db['box'])
        # The rows as the data has them, kept for the comparison and to go back.
        self.base = {c['row']: np.array(backend.pose[c['start']:c['stop']], dtype=np.float32) for c in self.clips}
        self.sources = {}
        self.roots = {}
        self.cache = {}
        self.applied = {'pick': '', 'place': ''}
        self.info = {'pick': None, 'place': None}
        self.last = None
        self.shownMemo = {}
        self.solveMs = 0.0
        backend.edits = self

    @property
    def available(self):
        return len(self.clips) > 0

    def source(self, clip):
        """The clip's rows as a pick editor source: qpos (T, 36) and box (T, 7) in a frame where the root starts at
        the origin, facing +x, in the order of a pick (a place backward)."""
        row = clip['row']
        if row in self.sources:
            return self.sources[row]
        L, W = self.backend.L, self.backend.width
        rows = self.base[row]
        T = clip['stop'] - clip['start']
        root = self.root_of(clip)

        def at(t, key, n):
            o = L[key][0]
            return [float(rows[t, o + i]) for i in range(n)]

        def world(t, p):
            r = root[t]
            c, s = math.cos(r['yaw']), math.sin(r['yaw'])
            return [r['pos'][0] + c * p[0] - s * p[1], r['pos'][1] + s * p[0] + c * p[1], p[2]]

        qpos, box = [], []
        for t in range(T):
            R = qfrom_yaw(root[t]['yaw'])
            q = np.zeros(36)  # (36,)
            q[0:3] = world(t, at(t, 'hip_pos', 3))
            q[3:7] = qabs(qnorm(qmul(R, at(t, 'hip_quat', 4))))
            q[7:36] = at(t, 'joints', 29)
            qpos.append(q)
            box.append(world(t, at(t, 'box_pos', 3)) + qabs(qnorm(qmul(R, at(t, 'box_quat', 4)))))
        place = clip['kind'] == 'place'
        if place:
            qpos.reverse()
            box.reverse()
        attach = T - clip['release'] if place else clip['attach']
        step = 2 if T % 2 == 1 else 1
        src = {
            'name': CLIP[clip['kind']] + (' mirrored' if clip['mirror'] else ''), 'fps': self.cfg['fps'] / step,
            'stop': 0, 'attach': math.ceil(attach / step), 'lifted': (T - 1) / step,
            'box_size_data': self.dataBox, 'box_size_physics': self.dataBox,
            'qpos': qpos[::step], 'box': box[::step],
        }
        out = {'src': src, 'root': root, 'qpos': qpos, 'box': box, 'step': step,
               'editor': PickEdit(self.model, src), 'T': T}
        self.sources[row] = out
        return out

    def root_of(self, clip):
        """The database's root per frame of a clip, {pos, yaw} in the frame of its first one."""
        row = clip['row']
        if row in self.roots:
            return self.roots[row]
        L, W = self.backend.L, self.backend.width
        rows = self.base[row]
        T = clip['stop'] - clip['start']
        k = L['root_delta'][0]
        root = []
        px = py = yaw = 0.0
        for t in range(T):
            root.append({'pos': [px, py, 0.0], 'yaw': yaw})
            d = [float(rows[t, k + i]) for i in range(3)]
            px += math.cos(yaw) * d[0] - math.sin(yaw) * d[1]
            py += math.sin(yaw) * d[0] + math.cos(yaw) * d[1]
            yaw += d[2]
        self.roots[row] = root
        return root

    def manual_of(self, kind):
        """The manual edit of the pick up or of the put down ('pick' | 'place')."""
        return self.manualPlace if kind == 'place' and self.manualPlace else self.manual

    def palm_range(self, hand, axis, box_size, as_='pick'):
        """The range (m) of a palm's shift on its side face, hand 0 left or 1 right, axis 0 along the face or 1 up it,
        for the box of the pick up or of the put down, as the pick editor limits it."""
        clip = (next((c for c in self.clips if c['kind'] == as_), None)
                or next((c for c in self.clips if c['kind'] == 'pick'), None))
        if not clip or not self.model:
            return [-0.1, 0.1]
        S = self.source(clip)
        editor = S['editor']
        c = editor.context(self.edit(0, box_size, as_))
        p = editor.palm_in_box(c, editor.attach, hand, {'size': 1, 'sym': 1})['p'][2 if axis else 0]
        half = c['half'][2 if axis else 0]
        return [-half + FACE_MARGIN - p, half - FACE_MARGIN - p]

    def edit(self, support, box_size, as_='pick'):
        """The edit for a box resting `support` m above the floor, with the manual edit of the pick up or of the put
        down (as_)."""
        s, m = self.settings, self.manual_of(as_)
        size = box_size if (s['autoSize'] and box_size
                            and any(abs(v - self.dataBox[i]) > SIZE_TOL for i, v in enumerate(box_size))) \
            else self.manual['box']
        up = support if s['autoHeight'] else m['up']
        return edit_of({
            'alignBox': False, 'forward': m['forward'], 'left': m['left'], 'yaw': m['yaw'],
            'up': up if up >= SUPPORT_MIN else 0,
            'grasp': m['grasp'], 'grip': {'keep': m['gripKeep']}, 'box': list(size) if size else None,
            'fitBox': True,
        })

    def key(self, edit):
        return json.dumps([edit['forward'], edit['left'], edit['yaw'], _to_fixed3(edit['up']),
                           edit['grasp'], edit['grip']['keep'], edit['box']])

    def solved(self, clip, edit):
        """The solved rows of a clip for an edit, cached. With no edit, no IK model or a failed solve, the data's
        rows (failed says why)."""
        k = '%d %s' % (clip['row'], self.key(edit))
        if k in self.cache:
            return self.cache[k]
        t0 = time.perf_counter()
        if is_no_edit(edit):
            out = self.unedited(clip, edit, None)
        elif not self.model:
            out = self.unedited(clip, edit, 'the IK model did not load')
        else:
            try:
                out = self.solve(clip, self.source(clip), edit)
            except Exception as e:
                out = self.unedited(clip, edit, str(e))
        out['ms'] = (time.perf_counter() - t0) * 1000.0
        self.solveMs += out['ms']
        self.cache[k] = out
        return out

    def unedited(self, clip, edit, failed):
        base = self.base[clip['row']]
        k = self.backend.L['hip_pos'][0] + 2
        low = float(np.min(base[:, k]))
        return {'rows': base, 'offset': [0, 0, 0], 'edit': edit, 'solved': None, 'pelvisMin': low,
                'basePelvisMin': low, 'palmMiss': None, 'failed': failed}

    def solve(self, clip, S, edit):
        src, root, editor, T, step = S['src'], S['root'], S['editor'], S['T'], S['step']
        L, W = self.backend.L, self.backend.width
        base = self.base[clip['row']]
        rows = base.copy()  # (T, W) float32
        res = editor.solve(edit)
        if len(res['qpos']) != len(src['qpos']):
            raise RuntimeError('a pick motion edit changed the number of frames')
        qpos = res['qpos'] if step == 1 else upsample(S['qpos'], res['qpos'], 3)
        box = res['box'] if step == 1 else upsample(S['box'], res['box'], 3)
        if clip['kind'] == 'place':
            qpos = list(reversed(qpos))
            box = list(reversed(box))
        fps = self.cfg['fps']

        def local(t, p):
            r = root[t]
            dx, dy = p[0] - r['pos'][0], p[1] - r['pos'][1]
            c, s = math.cos(r['yaw']), math.sin(r['yaw'])
            return [c * dx + s * dy, -s * dx + c * dy, p[2]]

        hip_pos, hip_rot, base_hip, base_rot = [], [], [], []
        for t in range(T):
            R = qfrom_yaw(root[t]['yaw'])
            hp = local(t, qpos[t])
            hr = qabs(qnorm(qinv_mul(R, list(qpos[t][3:7]))))
            hip_pos.append(hp)
            hip_rot.append(hr)
            base_hip.append([float(base[t, L['hip_pos'][0] + i]) for i in range(3)])
            base_rot.append([float(base[t, L['hip_quat'][0] + i]) for i in range(4)])
            rows[t, L['hip_pos'][0]:L['hip_pos'][0] + 3] = hp
            rows[t, L['hip_quat'][0]:L['hip_quat'][0] + 4] = hr
            rows[t, L['joints'][0]:L['joints'][0] + 29] = qpos[t][7:36]
            rows[t, L['box_pos'][0]:L['box_pos'][0] + 3] = local(t, box[t])
            rows[t, L['box_quat'][0]:L['box_quat'][0] + 4] = qabs(qnorm(qinv_mul(R, list(box[t][3:7]))))
        # Velocities: the data's, plus the change the edit makes to them, so frames the edit leaves alone keep theirs.
        for i in range(3):
            dv = central_diff([hip_pos[t][i] - base_hip[t][i] for t in range(T)], fps)  # (T,)
            col = L['hip_vel'][0] + i
            rows[:, col] = (rows[:, col].astype(np.float64) + dv).astype(np.float32)
        joints0 = L['joints'][0]
        for j in range(29):
            dv = central_diff([qpos[t][7 + j] - float(base[t, joints0 + j]) for t in range(T)], fps)  # (T,)
            col = L['joint_vel'][0] + j
            rows[:, col] = (rows[:, col].astype(np.float64) + dv).astype(np.float32)
        w_new = angular_velocity(unroll(hip_rot), fps)
        w_old = angular_velocity(unroll(base_rot), fps)
        for i in range(3):
            col = L['hip_ang'][0] + i
            d = np.array([w_new[t][i] - w_old[t][i] for t in range(T)])  # (T,)
            rows[:, col] = (rows[:, col].astype(np.float64) + d).astype(np.float32)
        miss = res['miss'][src['attach']] if src['attach'] < len(res['miss']) else None
        return {'rows': rows, 'offset': self.offset(editor, src, res), 'edit': edit, 'solved': res,
                'pelvisMin': min(p[2] for p in hip_pos), 'basePelvisMin': min(p[2] for p in base_hip),
                'palmMiss': miss['palm'] if miss else None, 'failed': None}

    def offset(self, editor, src, res):
        """How the edited box at rest lies against the data's: [forward, left, yaw] in the frame of the data's box
        at rest with its x axis turned to the robot's heading at the grasp, so forward is away from the robot."""
        a, b, h = src['box'][src['attach'] - 1], res['box'][src['attach'] - 1], editor.heading
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = qyaw(b[3:7]) - qyaw(a[3:7])
        turn = math.atan2(math.sin(d), math.cos(d))
        return [math.cos(h) * dx + math.sin(h) * dy, -math.sin(h) * dx + math.cos(h) * dy, turn]

    def use(self, kind, support, box_size, as_=None):
        """Puts the rows of the edit for a pick or a place in the Animation Database, for a box `support` m up, with
        the manual edit of as_ (the kind's own without it). Returns whether the rows changed."""
        edit = self.edit(support, box_size, as_ or kind)
        k = self.key(edit)
        if self.applied[kind] == k:
            return False
        W = self.backend.width
        info = None
        for clip in self.clips:
            if clip['kind'] != kind:
                continue
            r = self.solved(clip, edit)
            self.backend.pose[clip['start']:clip['stop']] = r['rows']
            if info is None or clip['mirror'] == 0:
                info = dict(r, clip=clip)
        self.applied[kind] = k
        self.info[kind] = info
        return True

    def note(self, supports, box_size):
        """What the last sync was asked for, which shown() reads."""
        self.last = {'supports': dict(supports), 'boxSize': list(box_size) if box_size else box_size}

    def sync(self, supports, box_size, bend=None):
        """The edits for the supports {pick, place} (m) and the page's box size. bend: the bend that plays ('pick' or
        'place'), whose height and manual edit both clips get, or None, where each has its own. Returns the kinds whose
        rows changed."""
        self.note(supports, box_size)
        out = []
        for kind in ('pick', 'place'):
            as_ = bend or kind
            if self.use(kind, supports.get(as_) or 0, box_size, as_):
                out.append(kind)
        return out

    def own_key(self, kind):
        """The key of a kind's own edit for the last sync, or None before one."""
        l = self.last
        return None if l is None else self.key(self.edit(l['supports'].get(kind) or 0, l['boxSize'], kind))

    def borrowed(self, kind):
        """The kind plays with the edit of the other kind's bend, not with its own."""
        k = self.own_key(kind)
        return k is not None and self.applied[kind] != '' and self.applied[kind] != k

    def shown(self, kind):
        """The pick or the place with its own edit, as info has it, for the Pick motion tab: while a bend plays, the
        other clip has the edit of that bend, and `borrowed` says so. Solves the kind's own edit if it is not cached."""
        l = self.last
        clip = (next((c for c in self.clips if c['kind'] == kind and c['mirror'] == 0), None)
                or next((c for c in self.clips if c['kind'] == kind), None))
        if l is None or clip is None:
            return self.info[kind]
        edit = self.edit(l['supports'].get(kind) or 0, l['boxSize'], kind)
        k = self.key(edit)
        memo = self.shownMemo.get(kind)
        if memo is None or memo['key'] != k:
            self.shownMemo[kind] = {'key': k, 'info': dict(self.solved(clip, edit), clip=clip)}
        return dict(self.shownMemo[kind]['info'], borrowed=self.applied[kind] != '' and self.applied[kind] != k)

    def set_manual(self, texts, box_size):
        """Sets the manual edit from texts, as the page's address gives them. rest: "forward,left,up,yaw" (cm, cm, cm,
        degrees). palms: "along,up" for both palms or "left along,left up,right along,right up" (cm). grip: "keep" or
        "data". A text of the put down (placeRest, placePalms, placeGrip) gives it an edit of its own, which starts as
        the pick's. A value outside its range is brought to the range."""
        def read(m, rest, palms, grip):
            if rest is not None:
                for k, v in zip(REST_KEYS, _numbers(rest, (4,), 'box at rest')):
                    lo, hi = REST_LIMITS[k]
                    c = min(hi, max(lo, v))
                    m[k] = c / DEG if k == 'yaw' else c / 100
            if palms is not None:
                v = _numbers(palms, (2, 4), 'palms')
                m['grasp'] = [[v[0] / 100, v[1] / 100], [v[2] / 100, v[3] / 100] if len(v) == 4 else [v[0] / 100, v[1] / 100]]
            if grip is not None:
                if grip not in ('keep', 'data'):
                    raise ValueError(f'bad grip "{grip}"')
                m['gripKeep'] = grip == 'keep'

        pick = own_manual(self.manual)
        read(pick, texts.get('rest'), texts.get('palms'), texts.get('grip'))
        place = own_manual(self.manualPlace) if self.manualPlace else None
        if any(texts.get(k) is not None for k in ('placeRest', 'placePalms', 'placeGrip')):
            if place is None:
                place = own_manual(pick)
            read(place, texts.get('placeRest'), texts.get('placePalms'), texts.get('placeGrip'))
        for as_, m in (('pick', pick), ('place', place)):
            if m is None:
                continue
            for h in range(2):
                for x in range(2):
                    r = self.palm_range(h, x, box_size, as_)
                    m['grasp'][h][x] = min(math.floor(200 * r[1]) / 200, max(math.ceil(200 * r[0]) / 200, m['grasp'][h][x]))
        # Nothing is set before every text was read, so a bad text leaves the edit as it was.
        self.manual.update(pick)
        self.manualPlace = place

    def manual_texts(self):
        """The texts of set_manual for the manual edit now, None where it is the default (the put down: the pick's)."""
        def num(v):
            return '%g' % (v + 0.0)

        def rest(m):
            return ','.join(num(_round1(m[k] * DEG if k == 'yaw' else 100 * m[k])) for k in REST_KEYS)

        def palms(m):
            v = [_round1(100 * x) for g in m['grasp'] for x in g]
            return ','.join(num(x) for x in (v[:2] if v[0] == v[2] and v[1] == v[3] else v))

        def grip(m):
            return 'keep' if m['gripKeep'] else 'data'

        m, pl, d = self.manual, self.manualPlace, MANUAL
        differs = lambda f: pl is not None and f(pl) != f(m)
        return {'rest': rest(m) if rest(m) != rest(d) else None, 'palms': palms(m) if palms(m) != palms(d) else None,
                'grip': 'keep' if m['gripKeep'] else None,
                'placeRest': rest(pl) if differs(rest) else None, 'placePalms': palms(pl) if differs(palms) else None,
                'placeGrip': grip(pl) if differs(grip) else None}

    def offsets(self):
        """The box at rest of the edited pick and place against the data's, [forward, left, yaw]."""
        z = [0, 0, 0]
        return {'pick': self.info['pick']['offset'] if self.info['pick'] else z,
                'place': self.info['place']['offset'] if self.info['place'] else z}

    def lift(self, g):
        """How much higher than the data the edited pelvis is at row g of the Animation Database."""
        c = self.clip_of(g)
        if not c or not self.info[c['kind']]:
            return 0.0
        k = self.backend.L['hip_pos'][0] + 2
        return float(self.backend.pose[g, k]) - float(self.base[c['row']][g - c['start'], k])

    def restore(self):
        """The data's rows back in the Animation Database."""
        for c in self.clips:
            self.backend.pose[c['start']:c['stop']] = self.base[c['row']]
        self.applied = {'pick': '', 'place': ''}
        self.info = {'pick': None, 'place': None}

    def prepare(self, supports, box_size, as_='pick'):
        """Solves ahead the edits of these supports, so a table the robot goes to does not wait for the solve."""
        for s in supports:
            edit = self.edit(s, box_size, as_)
            for clip in self.clips:
                self.solved(clip, edit)

    def clear_cache(self):
        self.cache.clear()
        self.applied = {'pick': '', 'place': ''}

    def clip_of(self, g):
        """The edited clip with row g of the Animation Database, or None."""
        return next((c for c in self.clips if c['start'] <= g < c['stop']), None)
