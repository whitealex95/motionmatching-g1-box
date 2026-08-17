"""Readers for the SceneBot clip bundle and contact labels.

clips.bin is one flat float32 array; clips_index.json holds per-clip element
offsets. Joint data is Isaac-order (29), body data is Isaac BFS body order
(30 bodies, pelvis first), body quaternions are wxyz, all at 50 Hz.
"""
import json

import numpy as np


def _clamp_frame(f, n):
    f = int(np.floor(f)) if np.isfinite(f) else 0
    return 0 if f < 0 else (max(0, n - 1) if f >= n else f)


class Clip:
    def __init__(self, floats, entry):
        self.name = entry['name']
        self.n_frames = int(entry['n_frames'])
        jd = int(entry['joint_dim'])
        bc = int(entry['body_count'])
        nf = self.n_frames
        o = int(entry['joint_pos_off'])
        self.joint_pos = floats[o:o + nf * jd].reshape(nf, jd)
        o = int(entry['joint_vel_off'])
        self.joint_vel = floats[o:o + nf * jd].reshape(nf, jd)
        o = int(entry['body_pos_w_off'])
        self.body_pos = floats[o:o + nf * bc * 3].reshape(nf, bc, 3)
        o = int(entry['body_quat_w_off'])
        self.body_quat = floats[o:o + nf * bc * 4].reshape(nf, bc, 4)  # wxyz

    def frame(self, f):
        return _clamp_frame(f, self.n_frames)


class ClipBundle:
    def __init__(self, bin_path, index_path):
        floats = np.fromfile(bin_path, np.float32)
        with open(index_path) as f:
            index = json.load(f)
        self._clips = {int(k): Clip(floats, v) for k, v in index.items()}

    def clip(self, idx):
        return self._clips[int(idx)]


class ContactLabels:
    def __init__(self, bin_path, index_path):
        self._floats = np.fromfile(bin_path, np.float32)
        with open(index_path) as f:
            self._index = {int(k): v for k, v in json.load(f).items()}

    def has_clip(self, idx):
        e = self._index.get(int(idx))
        return bool(e and e.get('present'))

    def at_frame(self, idx, f):
        e = self._index.get(int(idx))
        if not e or not e.get('present'):
            return None
        f = _clamp_frame(f, int(e['n_frames']))
        o = int(e['offset'])
        d = int(e['dim'])
        return self._floats[o + f * d:o + (f + 1) * d]

    def max_stream_dim(self):
        dim = 4
        for e in self._index.values():
            if e.get('present') and int(e['dim']) > dim:
                dim = int(e['dim'])
        return dim
