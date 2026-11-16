"""The pick motion (mm_g1/pickmotion.py) beside the control loop. One solve takes 0.6 to 1.7 s in Python, longer than
the stream's lookahead, so the box driver never solves on the control loop: the Sim asks ready() before it puts an
edit in the Animation Database, and an edit that is not solved yet is solved here while the motion that played goes on.
The lock is PickMotion's one lock (its module docstring): this thread holds it while it solves, and the control loop
skips the sync of a step it finds held."""

import math
import queue
import threading
import time

import numpy as np

from mm_g1.mathq import qfrom_yaw, qmul, qyaw
from mm_g1.pickmotion import SUPPORT_MIN
from mm_g1.sim import TABLE

VIEW_STEP = 2               # the viewer's clip at 30 fps, every second row of the 60 fps Animation Database


class PickWorker:
    def __init__(self, pm, say):
        self.pm, self.say = pm, say
        self.lock = threading.Lock()
        self.jobs = queue.Queue()
        self.asked = set()
        self.busy = None            # (support, box) being solved
        self.solved_n = 0
        self.ranges = None          # the palm shift ranges for the box solved last, [[lx, lz], [rx, rz]] as [lo, hi]
        threading.Thread(target=self.loop, daemon=True).start()

    def key(self, support, box_size):
        return self.pm.key(self.pm.edit(support, box_size))

    def ready(self, kind, support, box_size):
        """True when the edit of this kind for this support is solved. Else it is asked for and False."""
        pm = self.pm
        k = self.key(support, box_size)
        if all(f"{c['row']} {k}" in pm.cache for c in pm.clips if c['kind'] == kind):
            return True
        job = (round(float(support), 3), tuple(box_size))
        if (k, job) not in self.asked:
            self.asked.add((k, job))
            self.jobs.put((k, job))
        return False

    def loop(self):
        while True:
            k, (support, box) = self.jobs.get()
            self.busy = (support, list(box))
            t0 = time.perf_counter()
            with self.lock:
                self.pm.prepare([support], list(box))
                self.ranges = [[self.pm.palm_range(h, a, list(box)) for a in (0, 1)] for h in (0, 1)]
            self.asked.discard((k, (support, box)))
            self.busy = None
            self.solved_n += 1
            if time.perf_counter() - t0 > 0.1:
                self.say(f'pick motion: solved for a box {support:.2f} m above the floor in {time.perf_counter() - t0:.1f} s')

    def state(self, supports):
        """The Pick motion card's numbers: the settings, the supports, what plays for each kind and what is solving."""
        pm = self.pm
        out = {'settings': dict(pm.settings), 'manual': pm.manual, 'supports': dict(supports),
               'data_box': list(pm.dataBox), 'available': pm.available and pm.model is not None,
               'solving': None if self.busy is None else {'support': self.busy[0], 'box': self.busy[1]},
               'queued': self.jobs.qsize(), 'solved': self.solved_n, 'ranges': self.ranges, 'kinds': {}}
        for kind in ('pick', 'place'):
            info = pm.info[kind]
            out['kinds'][kind] = None if info is None else {
                'key': pm.applied[kind], 'edit': info['edit'], 'failed': info['failed'], 'solved': info['solved'] is not None,
                'pelvis_min': info['pelvisMin'], 'base_pelvis_min': info['basePelvisMin'],
                'palm_miss': None if info['palmMiss'] is None else list(info['palmMiss']), 'offset': list(info['offset']),
                'ms': round(info.get('ms', 0.0)), 'frames': info['clip']['stop'] - info['clip']['start']}
        return out

    def clip(self, kind):
        """The viewer's clip of a kind, as the page's PickView draws it: the edited motion and the data's in the frame of
        the clip's first root, MuJoCo qpos (36,) and box [pos, wxyz] per frame at 30 fps, and the table under the edited
        box at rest. None before the kind has played."""
        pm = self.pm
        info = pm.info[kind]
        if info is None:
            return None
        clip = info['clip']
        root = pm.root_of(clip)
        T = clip['stop'] - clip['start']
        base = pm.base[clip['row']]
        frames = list(range(0, T, VIEW_STEP))
        if frames[-1] != T - 1:
            frames.append(T - 1)
        now = [self.pose_at(info['rows'], root, t) for t in frames]
        data = [self.pose_at(base, root, t) for t in frames]
        size = info['edit']['box'] or pm.dataBox
        table = None
        up = info['edit']['up']
        if up >= SUPPORT_MIN:
            rest = self.pose_at(info['rows'], root, 0 if kind == 'pick' else T - 1)[1]
            yaw = qyaw(rest[3:7])
            reach = TABLE['size'][0] / 2 - TABLE['inset'] - size[0] / 2
            table = {'pos': [rest[0] + reach * math.cos(yaw), rest[1] + reach * math.sin(yaw)], 'yaw': yaw,
                     'size': [TABLE['size'][0], TABLE['size'][1], up]}
        r4 = lambda v: [round(float(x), 4) for x in v]
        return {'kind': kind, 'key': pm.applied[kind], 'fps': pm.cfg['fps'] / VIEW_STEP, 'size': list(size),
                'data_size': list(pm.dataBox), 'table': table,
                'qpos': [r4(q) for q, _ in now], 'box': [r4(b) for _, b in now],
                'data_qpos': [r4(q) for q, _ in data], 'data_box': [r4(b) for _, b in data]}

    def pose_at(self, rows, root, t):
        """(qpos (36,), box (7,)) at frame t of rows (T, W), in the frame of the clip's first root (PickView.poseAt)."""
        L = self.pm.backend.L
        r = root[t]
        R = qfrom_yaw(r['yaw'])
        c, s = math.cos(r['yaw']), math.sin(r['yaw'])
        at = lambda k, n: [float(x) for x in rows[t, L[k][0]:L[k][0] + n]]
        world = lambda p: [r['pos'][0] + c * p[0] - s * p[1], r['pos'][1] + s * p[0] + c * p[1], p[2]]
        q = np.zeros(36)  # (36,)
        q[0:3] = world(at('hip_pos', 3))
        q[3:7] = qmul(R, at('hip_quat', 4))
        q[7:36] = at('joints', 29)
        return q, [*world(at('box_pos', 3)), *qmul(R, at('box_quat', 4))]
