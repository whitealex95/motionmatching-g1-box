"""Port of lmm-g1-box's web/js/pickmotion.js: the pick motion that plays. The SceneBot pick and place of the data
variant, solved again by the pick editor's whole-body IK (pickedit.py) for the box the robot goes for. The Matching
Databases, the costs and the search stay as they are. Only the rows of the Animation Database that play these clips
change: the pelvis, the joints, the box and their velocities. The root, the contacts and the hold state stay the
data's.

The source of an edit is the clip as the database has it, at 60 fps, from the frame where the walk ends. A place is
solved as a pick: its rows played backward are a pick, and the solved frames go back in the place's order.

Threads: an instance has no global state and may run off the control loop, but it is not itself thread safe. One
lock has to cover
  * every call that writes backend.pose, that is use(), sync() and restore(), against the control loop reading it,
  * solved(), prepare() and palm_range(), which share the one BodyModel and the caches (cache, sources, roots),
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
        self.dataBox = list(cfg['box_size'])
        self.clips = scene_bot_clips(cfg, backend.db['box'])
        # The rows as the data has them, kept for the comparison and to go back.
        self.base = {c['row']: np.array(backend.pose[c['start']:c['stop']], dtype=np.float32) for c in self.clips}
        self.sources = {}
        self.roots = {}
        self.cache = {}
        self.applied = {'pick': '', 'place': ''}
        self.info = {'pick': None, 'place': None}
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

    def palm_range(self, hand, axis, box_size):
        """The range (m) of a palm's shift on its side face, as the pick editor limits it."""
        clip = next((c for c in self.clips if c['kind'] == 'pick'), None)
        if not clip or not self.model:
            return [-0.1, 0.1]
        S = self.source(clip)
        editor = S['editor']
        c = editor.context(self.edit(0, box_size))
        p = editor.palm_in_box(c, editor.attach, hand, {'size': 1, 'sym': 1})['p'][2 if axis else 0]
        half = c['half'][2 if axis else 0]
        return [-half + FACE_MARGIN - p, half - FACE_MARGIN - p]

    def edit(self, support, box_size):
        """The edit of a pick or a place for a box resting `support` m above the floor."""
        s, m = self.settings, self.manual
        size = box_size if (s['autoSize'] and box_size
                            and any(abs(v - self.dataBox[i]) > SIZE_TOL for i, v in enumerate(box_size))) \
            else m['box']
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

    def use(self, kind, support, box_size):
        """Puts the rows of the edit for a pick or a place in the Animation Database. Returns whether they changed."""
        edit = self.edit(support, box_size)
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

    def sync(self, supports, box_size):
        """The edits for the supports {pick, place} (m) and the page's box size. Returns the kinds that changed."""
        return [kind for kind in ('pick', 'place') if self.use(kind, supports.get(kind) or 0, box_size)]

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

    def prepare(self, supports, box_size):
        """Solves ahead the edits of these supports, so a table the robot goes to does not wait for the solve."""
        for s in supports:
            edit = self.edit(s, box_size)
            for clip in self.clips:
                self.solved(clip, edit)

    def clear_cache(self):
        self.cache.clear()
        self.applied = {'pick': '', 'place': ''}

    def clip_of(self, g):
        """The edited clip with row g of the Animation Database, or None."""
        return next((c for c in self.clips if c['start'] <= g < c['stop']), None)
