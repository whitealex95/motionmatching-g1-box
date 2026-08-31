"""Per-clip label sidecars: <clip>.labels.yaml next to the clip's .npz.

Label resolution for a box clip, per key, most specific first:
  1. the clip's sidecar file (hand-tunable YAML, committed to git)
  2. the box-height rule (boxes.segment_phases) -- what the sidecar
     generator (tools/make_labels.py) materializes, so tuning always
     starts from the rule's actual output

Schema (frames at the clip's own fps, half-open [start, stop) spans):

    phases:                # must tile [0, frames)
      pick:  [0, 132]
      carry: [132, 411]
      place: [411, 520]
    contact:               # per-hand ON spans; attach is DERIVED as the
      left_hand:  [[125, 466]]     # union span of all contact intervals
      right_hand: [[125, 466]]

Omit a key to keep the rule's value for it. Sidecar contents are hashed
into the bake fingerprint, so edits rebuild the cache automatically.
"""
import os

import numpy as np
import yaml

from . import boxes
from .states import Phase

_HAND_CH = {"left_hand": 2, "right_hand": 3}


def sidecar_path(data_dir, stem):
    return os.path.join(data_dir, stem + ".labels.yaml")


def _spans_to_mask(spans, T):
    m = np.zeros(T, bool)
    for a, b in spans:
        m[max(0, int(a)):min(T, int(b))] = True
    return m


def box_labels(stem, box_pose, data_dir):
    """(phase (T,), attach (T,) bool, contact (T,5)) for one box clip."""
    T = len(box_pose)
    phase, attach, _info = boxes.segment_phases(box_pose[:, 0:3])

    path = sidecar_path(data_dir, stem)
    sc = None
    if os.path.exists(path):
        with open(path) as f:
            sc = yaml.safe_load(f) or {}
        if "phases" in sc:
            phase = np.empty(T, np.int32)
            spans = sc["phases"]
            for key, code in (("pick", Phase.PICK), ("carry", Phase.CARRY),
                              ("place", Phase.PLACE)):
                a, b = spans[key]
                phase[int(a):int(b)] = code

    contact = np.zeros((T, 5))
    if sc is not None and "contact" in sc:
        for hand, ch in _HAND_CH.items():
            contact[:, ch] = _spans_to_mask(sc["contact"].get(hand, []), T)
        # Attach = box rides the robot: the collapsed span of hand contact.
        on = np.flatnonzero(contact[:, 2:4].any(1))
        attach = np.zeros(T, bool)
        if len(on):
            attach[on[0]:on[-1] + 1] = True
    else:
        # No recorded labels: wrists prompt object contact while the box is
        # attached, feet/pelvis stay zero (the demo's plain-walking pattern).
        contact[:, 2] = contact[:, 3] = attach.astype(float)
    return phase, attach, contact
