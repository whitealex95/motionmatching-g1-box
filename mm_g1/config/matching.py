"""Motion-matching search settings: features, springs, search biases, weights."""
from .library import HORIZONS

# --- GenoView (Holden "Simple Motion Matching") heuristics ---------------------------
# All math + hyperparameters mirror ../GenoViewPython-MotionMatching/genoview_g1.py.
SEARCH_TIME = 0.15             # seconds between database searches
INERT_HALFLIFE = 0.075         # inertialization (pose-transition) blend half-life
VEL_HALFLIFE = 0.2             # desired-trajectory position spring half-life
ROT_HALFLIFE = 0.2             # desired-trajectory rotation spring half-life
CURRENT_BIAS = 0.01            # stay-in-clip bias seeded onto the current frame's distance
APPROX_BIAS = 0.01             # cKDTree eps: slightly approximate (faster) nearest-neighbour

# Savitzky-Golay windows for the smoothed "simulation root" (genoview's 31/61 @60fps,
# time-matched to our 30 fps). Smooths per-step bob/sway out of the root the matcher tracks.
ROOT_POS_SMOOTH = 15
ROOT_DIR_SMOOTH = 31

# GenoView trims the last HORIZONS[-1] frames of each clip from the SEARCH only
# (cKDTree(X[rs:re-30])): the tail still plays out, but a match never lands there, so a
# full future trajectory always exists and the playhead can't run off the clip end.
SEARCH_TAIL = HORIZONS[-1]   # frames excluded from each clip's KD-tree (1.0 s @30fps)

# --- Box search-feature block weights, per phase ------------------------------------
# The planar box position / orientation blocks (expressed in the robot's gravity-aligned
# base frame). Scaled like the genoview blocks (one shared std per block, over that
# phase's own frames) then divided by these, so a heavier block contributes more to that
# phase's L2 distance. Each searchable box database names its own pair: no shared
# default, so a new database has to state what it wants.
# Ordered along the state-machine chain: pick -> carry -> place.

# PICK: when you press B the entry should be chosen mostly by where the box sits (and
# which way it faces) relative to the robot, even if the body pose matches a little worse.
# The pose pop is inertialized away, but a bad box placement is visible. With the single
# SceneBot pick, the ROT weight does the real work: all 16 entries sit at the same recorded
# stance-to-box offset (xy spread < 3 mm), so the position block is near-constant across
# candidates and barely moves the argmin. It goes live again if the OmniRetarget clips'
# own pick/place phases are re-enabled, where entries differ in stance-to-box offset.
PICK_BOX_POS_WEIGHT = 5.0
PICK_BOX_ROT_WEIGHT = 5.0

# CARRY has no box weights: its database is box-agnostic (features.build_db gives it the
# same 27-D pose + trajectory space as loco), so a WASD command steers the carry exactly
# the way it steers locomotion. The box still rides the robot, it is just not matched on.

# PLACE is entry-matched then ridden, like pick. Its 16 entries are 2 box-yaw folds x 8
# adjacent frames, and the box is ALREADY in the hands, so the position block varies by
# < 3 mm across candidates and is effectively inert: the rot weight is what picks the fold.
# Kept as its own pair for symmetry and for when multi-clip place data is re-enabled.
PLACE_BOX_POS_WEIGHT = 1.0
PLACE_BOX_ROT_WEIGHT = 5.0

BOX_INERT_HALFLIFE = 0.1  # box pose-transition (attach) inertialization half-life
