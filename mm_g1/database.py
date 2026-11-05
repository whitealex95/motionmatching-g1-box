"""The Matching Databases and the Animation Database of lmm-g1-box (web/js/controller.js: loadMMData, boxPart,
loadExtra, buildLoco, normalizedDb, lastSteps and the MMBackend), read from the copies in data/.

The page keeps features in float32 and computes costs in float64. The search here takes the squared distance from
a matrix-vector product, |q|^2 - 2 q.x + |x|^2 in float64, which differs from the page's running sum in the last
bits only."""
import json
import math
import os

import numpy as np

from .mathq import add, qfrom_yaw, qmul, qmul_vec, qnorm, qslerp_shortest_approx

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
VARIANT_DIR = os.path.join(DATA, 'sbplace')
EMM_LOCO_DIR = os.path.join(DATA, 'emm_loco')
EMM_CARRY_DIR = os.path.join(DATA, 'emm_carry')
ROBOT_JSON = os.path.join(DATA, 'robot.json')

NDOF = 29
NPOSE_FEATURES = 15
MIN_LIFT, LATE_STEP = 8, 30
ENTRY_LEAD = 30


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_half(path):
    return np.fromfile(path, dtype='<f2').astype(np.float32)


class MatchingDb:
    """X (N, n) float32, idx (N,) rows of the Animation Database, ranges (R, 2), range_stop (N,)."""

    def __init__(self, X, idx, ranges, offset=None, scale=None, range_kind=None):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.idx = np.asarray(idx, dtype=np.int64)
        self.ranges = np.asarray(ranges, dtype=np.int64).reshape(-1, 2)
        self.nfeatures = self.X.shape[1]
        self.range_stop = np.zeros(len(self.idx), dtype=np.int64)
        for a, b in self.ranges:
            self.range_stop[a:b] = b
        self.offset, self.scale = offset, scale
        self.range_kind = range_kind
        self.X64 = self.X.astype(np.float64)
        self.xx = np.einsum('ij,ij->i', self.X64, self.X64)
        self._allowed = {}
        self._gained = {}

    def allowed(self, ignore_range_end, entry):
        key = (ignore_range_end, entry)
        if key not in self._allowed:
            out = np.zeros(len(self.idx), dtype=bool)
            for r, (a, b) in enumerate(self.ranges):
                if entry and self.range_kind is not None and self.range_kind[r] != 1:
                    continue
                stop = b - ignore_range_end
                if entry:
                    stop = min(stop, a + entry)
                if stop > a:
                    out[a:stop] = True
            self._allowed[key] = out
        return self._allowed[key]

    def costs(self, qn, gain):
        """sum_j gain_j (qn_j - X_ij)^2 for every frame i."""
        q = qn.astype(np.float64)
        if np.all(gain == 1.0):
            return (q @ q) - 2.0 * (self.X64 @ q) + self.xx
        key = gain.tobytes()
        if key not in self._gained:
            Xg = self.X64 * np.sqrt(gain)
            self._gained = {key: (Xg, np.einsum('ij,ij->i', Xg, Xg))}
        Xg, xx = self._gained[key]
        qg = q * np.sqrt(gain)
        return (qg @ qg) - 2.0 * (Xg @ qg) + xx


def seq_sum(a):
    """The JS running sum over axis 0 (np.cumsum adds in order, a plain sum adds pairwise)."""
    return np.cumsum(a, axis=0)[-1]


def normalized_db(raw, idx, ranges, c):
    """A Matching Database of raw features (len, n), normalized on its own frames with the weights of c, as
    normalizedDb of the page (normalize_groups of lmm/database.py)."""
    n, length = raw.shape[1], raw.shape[0]
    offset = seq_sum(raw) / length
    scale = np.ones(n)
    j0 = 0
    for g, size in enumerate(c['group_sizes']):
        std = 0.0
        for j in range(j0, j0 + size):
            std += math.sqrt(float(seq_sum((raw[:, j] - offset[j]) ** 2)) / length) / size
        scale[j0:j0 + size] = std / c['weights'][g] if std > 1e-6 else 1.0
        j0 += size
    X = ((raw - offset) / scale).astype(np.float32)
    return MatchingDb(X, idx, ranges, offset=offset, scale=scale)


def last_steps(cfg, pose):
    """lastSteps of the page: per clip row the last step before its grasp or release, and `planted`, the frames of
    the Animation Database from a last step to the first step after it."""
    W, C = cfg['pose_width'], cfg['pose_layout']['contacts'][0]
    planted = np.zeros(cfg['nframes'], dtype=np.uint8)
    contacts = pose.reshape(-1, W)[:, C:C + 2]
    clips = []
    for row in cfg['clips']['rows']:
        start, stop, attach, release = row[2], row[3], row[4], row[5]
        T = stop - start
        down = ((contacts[start:stop, 0] > 0.5) & (contacts[start:stop, 1] > 0.5)).astype(np.uint8)
        t = 1
        while t < T:
            u = t
            while u < T and not down[u]:
                u += 1
            if t < u < T and u - t < MIN_LIFT and down[t - 1]:
                down[t:u] = 1
            t = u + 1
        out = []
        for kind, e in (('pick', attach), ('place', release)):
            if e is None or not (0 <= e < T):
                continue
            a = e
            if not down[e]:
                while a < min(T, e + LATE_STEP) and not down[a]:
                    a += 1
                if a >= T or not down[a]:
                    continue
            while a > 0 and down[a - 1]:
                a -= 1
            b = max(a, e)
            while b < T and down[b]:
                b += 1
            out.append({'kind': kind, 'event': e, 'step': a, 'lift': b})
            planted[start + a:start + b] = 1
        clips.append(out)
    return clips, planted


def load_db_file(path, c):
    buf = np.fromfile(path, dtype=np.uint8)
    nX = c['nframes'] * c['nfeatures'] * 4
    X = buf[:nX].view('<f4').reshape(c['nframes'], c['nfeatures'])
    idx = buf[nX:nX + c['nframes'] * 4].view('<i4')
    ranges = buf[nX + c['nframes'] * 4:nX + c['nframes'] * 4 + c['nranges'] * 8].view('<i4').reshape(-1, 2)
    return X, idx, ranges


def box_part(d, c, kind):
    """The pick (kind 0) or place (kind 1) ranges of the box database as a Matching Database of their own."""
    keep = [r for r in range(c['nranges']) if c['range_kind'][r] == kind]
    off, sc = np.asarray(c['offset'], np.float64), np.asarray(c['scale'], np.float64)
    raws, idxs, ranges, i = [], [], [], 0
    for r in keep:
        a, b = d.ranges[r]
        raws.append(d.X[a:b].astype(np.float64) * sc + off)
        idxs.append(d.idx[a:b])
        ranges.append((i, i + b - a))
        i += b - a
    return normalized_db(np.concatenate(raws), np.concatenate(idxs), ranges, c)


class Extra:
    """The EMM locomotion takes or the carry spans of EMM's box take: frames after the variant's in the Animation
    Database (loadExtra of the page)."""

    def __init__(self, folder, cfg, first):
        self.cfg = read_json(os.path.join(folder, 'config.json'))
        if self.cfg['nfeatures'] != cfg['controllers']['loco']['nfeatures'] or self.cfg['pose_width'] != cfg['pose_width']:
            raise ValueError(f'{folder} does not fit this variant')
        self.pose = load_half(os.path.join(folder, 'mm_pose.bin'))
        n = self.cfg['nfeatures']
        self.feats = np.fromfile(os.path.join(folder, 'mm_loco.bin'), dtype='<f4').reshape(-1, n)
        self.first = first


def loco_parts(name):
    return ('emm' if name in ('emm', 'emmboth') else 'nsm', 'emm' if name in ('emmcarry', 'emmboth') else 'nsm')


class MMBackend:
    """Exact Motion Matching over the databases (MMBackend of the page) with the loading of loadMMData and loadLoco."""

    kind = 'mm'

    def __init__(self, cfg=None, folder=VARIANT_DIR, loco='emmboth', emm_dir=EMM_LOCO_DIR, carry_dir=EMM_CARRY_DIR):
        self.cfg = cfg or read_json(os.path.join(folder, 'config.json'))
        cfg = self.cfg
        self.width = cfg['pose_width']
        self.L = cfg['pose_layout']
        pose = load_half(os.path.join(folder, 'mm_pose.bin'))
        self.ctrl = {}
        for name in ('loco', 'box'):
            c = cfg['controllers'][name]
            self.ctrl[name] = dict(c, offset=np.asarray(c['offset'], np.float32), scale=np.asarray(c['scale'], np.float32))
        self.db = {}
        for name in ('loco', 'box'):
            X, idx, ranges = load_db_file(os.path.join(folder, f'mm_{name}.bin'), cfg['controllers'][name])
            self.db[name] = MatchingDb(X, idx, ranges)
        self.db['box'].range_kind = list(cfg['controllers']['box']['range_kind'])
        self.db['pick'] = box_part(self.db['box'], cfg['controllers']['box'], 0)
        self.db['place'] = box_part(self.db['box'], cfg['controllers']['box'], 1)
        for name in ('pick', 'place'):
            self.ctrl[name] = dict(self.ctrl['box'], offset=self.db[name].offset, scale=self.db[name].scale)
        self.locos = {'nsm': self.db['loco']}
        self.loco_scale = {'nsm': (self.ctrl['loco']['offset'], self.ctrl['loco']['scale'])}
        self.loco = 'nsm'
        self.active = 'loco'
        self.frame = 0
        self.frac = 0.0
        self.steps, self.planted_rows = last_steps(cfg, pose)
        self.emm = self.emm_carry = None
        empty, carry = loco_parts(loco)
        parts = [pose]
        nframes = len(pose) // self.width
        if empty == 'emm':
            self.emm = Extra(emm_dir, cfg, nframes)
            parts.append(self.emm.pose)
            nframes += len(self.emm.pose) // self.width
        if carry == 'emm':
            self.emm_carry = Extra(carry_dir, cfg, nframes)
            parts.append(self.emm_carry.pose)
            nframes += len(self.emm_carry.pose) // self.width
        self.pose = np.concatenate(parts).reshape(-1, self.width)
        planted = np.zeros(len(self.pose), dtype=np.uint8)
        planted[:len(self.planted_rows)] = self.planted_rows
        self.planted_rows = planted
        self._masks = {}
        if loco != 'nsm':
            self.build_loco(loco)

    # ---- the Locomotion Matching Database (buildLoco, useLoco) ----

    def build_loco(self, name):
        empty, carry = loco_parts(name)
        c = self.cfg['controllers']['loco']
        n, tag, d = c['nfeatures'], c['nfeatures'] - 1, self.locos['nsm']
        off, sc = np.asarray(c['offset'], np.float64), np.asarray(c['scale'], np.float64)
        raws, idxs, ranges, i = [], [], [], 0
        for held in (True, False):
            extra = (self.emm_carry if carry == 'emm' else None) if held else (self.emm if empty == 'emm' else None)
            if extra is not None:
                for row in extra.cfg['clips']['rows']:
                    start, stop = row[2], row[3]
                    raws.append(extra.feats[start:stop].astype(np.float64))
                    idxs.append(extra.first + np.arange(start, stop))
                    ranges.append((i, i + stop - start))
                    i += stop - start
                continue
            for a, b in d.ranges:
                if (float(d.X[a, tag]) * float(sc[tag]) + float(off[tag]) > 0.5) != held:
                    continue
                raws.append(d.X[a:b].astype(np.float64) * sc + off)
                idxs.append(d.idx[a:b])
                ranges.append((i, i + b - a))
                i += b - a
        db = normalized_db(np.concatenate(raws), np.concatenate(idxs), ranges, c)
        self.locos[name] = db
        self.loco_scale[name] = (db.offset, db.scale)

    def use_loco(self, name):
        self.db['loco'] = self.locos[name]
        off, sc = self.loco_scale[name]
        self.ctrl['loco'] = dict(self.ctrl['loco'], offset=off, scale=sc)
        self.loco = name
        if self.active == 'loco':
            self.active = None

    # ---- features ----

    def normalize(self, name, query):
        c = self.ctrl[name]
        return ((query.astype(np.float64) - c['offset']) / c['scale']).astype(np.float32)

    def denormalize(self, name, x):
        c = self.ctrl[name]
        return (x.astype(np.float64) * c['scale'] + c['offset']).astype(np.float32)

    def current_features(self):
        d = self.db[self.active]
        a = d.X[self.frame]
        if not self.frac:
            return a
        b = d.X[self.frame + 1]
        a64 = a.astype(np.float64)
        return (a64 + self.frac * (b.astype(np.float64) - a64)).astype(np.float32)

    def end_of_anim(self):
        d = self.db[self.active]
        return self.frame + 1 >= d.range_stop[self.frame]

    def planted(self, name, frame):
        return name != 'loco' and self.planted_rows[self.db[name].idx[frame]] == 1

    # ---- per-frame masks of a box Matching Database ----

    def range_clip(self, name, r):
        d, rows = self.db[name], self.cfg['clips']['rows']
        g0 = d.idx[d.ranges[r][0]]
        lo, hi = 0, len(rows) - 1
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if rows[mid][2] <= g0:
                lo = mid
            else:
                hi = mid - 1
        nm, _, start, stop, attach, release = rows[lo]
        is_place = d.range_kind[r] if d.range_kind is not None else name == 'place'
        event = release if is_place else attach
        return self.cfg['clips']['names'][nm], start, stop, (-1 if event is None else event)

    def _mask(self, kind, name, make):
        key = (kind, name)
        if key not in self._masks:
            self._masks[key] = make(name)
        return self._masks[key]

    def no_entry(self, name):
        def make(name):
            d, out, sbn = self.db[name], np.zeros(len(self.db[name].idx), dtype=np.uint8), self.cfg.get('scenebot_entry_frames')
            for r, (a, b) in enumerate(d.ranges):
                g0, g1 = d.idx[a], d.idx[b - 1]
                clip, start, stop, event = self.range_clip(name, r)
                if g0 >= stop or not event >= 0:
                    continue
                if g1 - start >= event:
                    if sbn and clip.startswith('SceneBot'):
                        i = a + sbn
                        while i < b and d.idx[i] - start < event:
                            out[i] = 1
                            i += 1
                    continue
                first = a
                while first < b and not self.planted_rows[d.idx[first]]:
                    first += 1
                if first < b:
                    out[max(a, first - ENTRY_LEAD):b] = 1
            return out
        return self._mask('no_entry', name, make)

    def after_event(self, name):
        def make(name):
            d, out = self.db[name], np.zeros(len(self.db[name].idx), dtype=np.uint8)
            for r, (a, b) in enumerate(d.ranges):
                _, start, _, event = self.range_clip(name, r)
                if not event >= 0:
                    continue
                out[a:b] = (d.idx[a:b] - start >= event)
            return out
        return self._mask('after_event', name, make)

    def scenebot_pre(self, name):
        def make(name):
            d, out = self.db[name], np.zeros(len(self.db[name].idx), dtype=np.uint8)
            for r, (a, b) in enumerate(d.ranges):
                clip, start, _, event = self.range_clip(name, r)
                if not clip.startswith('SceneBot') or not event >= 0:
                    continue
                out[a:b] = (d.idx[a:b] - start < event)
            return out
        return self._mask('scenebot_pre', name, make)

    def before_event_mask(self, name):
        def make(name):
            out = self.no_entry(name).copy()
            out[self.after_event(name) == 0] = 1
            return out
        return self._mask('before_event', name, make)

    def walk_in_mask(self, name):
        def make(name):
            d, ne = self.db[name], self.no_entry(name)
            out = ne.copy()
            kind = (lambda r: d.range_kind[r]) if d.range_kind is not None else (lambda r: 1 if name == 'place' else 0)
            walks = [False, False]
            for r, (a, b) in enumerate(d.ranges):
                if ne[b - 1]:
                    walks[kind(r)] = True
            for r, (a, b) in enumerate(d.ranges):
                if walks[kind(r)]:
                    out[a:b][self.planted_rows[d.idx[a:b]] == 1] = 1
            return out
        return self._mask('walk_in', name, make)

    # ---- search and playback ----

    def match(self, name, qn, gain, ignore_range_end=20, ignore_surrounding=20, entry=0, mask=None, transition_cost=0.0):
        d = self.db[name]
        cur = self.frame if name == self.active and not self.end_of_anim() else -1
        best, best_cost = cur, math.inf
        if cur >= 0:
            f = self.current_features().astype(np.float64)
            best_cost = 0.0
            q64 = qn.astype(np.float64)
            for j in range(d.nfeatures):
                best_cost += gain[j] * (q64[j] - f[j]) ** 2
        cand = d.allowed(ignore_range_end, entry)
        if mask is not None or cur >= 0:
            cand = cand.copy()
            if mask is not None:
                cand &= mask == 0
            if cur >= 0:
                cand[max(0, cur - ignore_surrounding + 1):cur + ignore_surrounding] = False
        if cand.any():
            costs = transition_cost + d.costs(qn, gain)
            costs[~cand] = math.inf
            i = int(np.argmin(costs))
            if costs[i] < best_cost:
                best, best_cost = i, float(costs[i])
        cost = math.sqrt(max(0.0, best_cost if best == cur else best_cost - transition_cost))
        return {'name': name, 'frame': best, 'cost': cost, 'features': d.X[best], 'same': best == cur}

    def commit(self, m):
        self.active = m['name']
        self.frame = m['frame']
        self.frac = 0.0

    def row(self, name, frame):
        return self.pose[self.db[name].idx[frame]]

    def row_pose(self, r, root_pos, root_rot):
        L = self.L
        p = Pose()
        p.hipPos = r[L['hip_pos'][0]:L['hip_pos'][0] + 3].astype(np.float64).tolist()
        p.hipRot = qnorm(r[L['hip_quat'][0]:L['hip_quat'][0] + 4].astype(np.float64).tolist())
        p.hipVel = r[L['hip_vel'][0]:L['hip_vel'][0] + 3].astype(np.float64).tolist()
        p.hipAng = r[L['hip_ang'][0]:L['hip_ang'][0] + 3].astype(np.float64).tolist()
        p.joints = r[L['joints'][0]:L['joints'][0] + NDOF].astype(np.float64)
        p.jointVel = r[L['joint_vel'][0]:L['joint_vel'][0] + NDOF].astype(np.float64)
        rv, ra = L['root_vel'][0], L['root_ang'][0]
        p.rootVel = qmul_vec(root_rot, [float(r[rv]), float(r[rv + 1]), 0.0])
        p.rootAng = qmul_vec(root_rot, [0.0, 0.0, float(r[ra + 2])])
        p.rootPos = list(root_pos)
        p.rootRot = list(root_rot)
        c = L['contacts'][0]
        p.contacts = [bool(r[c] > 0.5), bool(r[c + 1] > 0.5)]
        p.held = float(r[L['held'][0]])
        p.boxPos = r[L['box_pos'][0]:L['box_pos'][0] + 3].astype(np.float64).tolist()
        p.boxRot = qnorm(r[L['box_quat'][0]:L['box_quat'][0] + 4].astype(np.float64).tolist())
        return p

    def match_pose(self, m, root_pos, root_rot, dt, speed=1.0):
        return self.scale_velocities(self.row_pose(self.row(m['name'], m['frame']), root_pos, root_rot), speed)

    def step(self, dt):
        pass

    def current_pose(self, root_pos, root_rot, dt, speed=1.0):
        L = self.L
        rd = L['root_delta'][0]
        pos, rot, left = root_pos, root_rot, speed
        while left > 1e-9:
            prev = self.row(self.active, self.frame)
            s = min(left, 1 - self.frac)
            pos = add(pos, qmul_vec(rot, [s * float(prev[rd]), s * float(prev[rd + 1]), 0.0]))
            rot = qnorm(qmul(qfrom_yaw(s * float(prev[rd + 2])), rot))
            left -= s
            if self.end_of_anim():
                self.frac = 0.0
                break
            self.frac += s
            if self.frac > 1 - 1e-9:
                self.frame += 1
                self.frac = 0.0
        p = self.row_pose(self.row(self.active, self.frame), pos, rot)
        if speed == 1 and not self.frac:
            return p
        t = self.frac
        if t:
            n = self.row_pose(self.row(self.active, self.frame + 1), pos, rot)
            lerp = lambda a, b: [x + t * (y - x) for x, y in zip(a, b)]
            p.hipPos = lerp(p.hipPos, n.hipPos)
            p.hipRot = qslerp_shortest_approx(p.hipRot, n.hipRot, t)
            p.boxPos = lerp(p.boxPos, n.boxPos)
            p.boxRot = qslerp_shortest_approx(p.boxRot, n.boxRot, t)
            p.held += t * (n.held - p.held)
            if t >= 0.5:
                p.contacts = n.contacts
            p.joints = p.joints + t * (n.joints - p.joints)
            p.jointVel = p.jointVel + t * (n.jointVel - p.jointVel)
            for k in ('hipVel', 'hipAng', 'rootVel', 'rootAng'):
                setattr(p, k, lerp(getattr(p, k), getattr(n, k)))
        return self.scale_velocities(p, speed)

    def scale_velocities(self, p, speed):
        if speed == 1:
            return p
        p.jointVel = p.jointVel * speed
        for k in ('hipVel', 'hipAng', 'rootVel', 'rootAng'):
            setattr(p, k, [v * speed for v in getattr(p, k)])
        return p

    # ---- the replan's snapshot: the playhead and the database in use, never the databases ----

    def state(self):
        return (self.active, self.frame, self.frac, self.loco)

    def set_state(self, s):
        active, self.frame, self.frac, loco = s
        if loco != self.loco:
            self.use_loco(loco)
        self.active = active


class Pose:
    __slots__ = ('rootPos', 'rootRot', 'rootVel', 'rootAng', 'hipPos', 'hipRot', 'hipVel', 'hipAng',
                 'joints', 'jointVel', 'contacts', 'held', 'boxPos', 'boxRot')

    def __init__(self):
        self.rootPos, self.rootRot, self.rootVel, self.rootAng = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        self.hipPos, self.hipRot, self.hipVel, self.hipAng = [0.0, 0.0, 0.77], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        self.joints, self.jointVel = np.zeros(NDOF), np.zeros(NDOF)
        self.contacts, self.held = [True, True], 0.0
        self.boxPos, self.boxRot = [0.3, 0.0, 0.7], [1.0, 0.0, 0.0, 0.0]

    def copy(self):
        p = Pose.__new__(Pose)
        for k in Pose.__slots__:
            v = getattr(self, k)
            setattr(p, k, v.copy() if isinstance(v, (list, np.ndarray)) else v)
        return p
