"""Configuration, split by concern:

    paths.py          repo-relative data / scene / cache locations
    library.py        WHAT the motion library contains and how it is baked
                      (clips, trims, folds, skill segmentation, box geometry)
    matching.py       motion-matching search: features, springs, biases, weights
    state_machine.py  the B-driven pick/carry/place behavior: command speeds,
                      the move-to-pick approach, box spawn

The low-level tracking controller (SONIC PD gains, joint orderings, rates)
is configured separately in tools/sonic_tracking/params.py.

All names are re-exported flat, so `from mm_g1 import config as C` and
`C.<NAME>` keep working; the split is about where a knob LIVES when reading.
"""
from .paths import *            # noqa: F401,F403
from .library import *          # noqa: F401,F403
from .matching import *        # noqa: F401,F403
from .state_machine import *   # noqa: F401,F403
