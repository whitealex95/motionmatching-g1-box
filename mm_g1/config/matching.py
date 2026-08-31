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

# Box search-feature block weights (box pos / orientation / linear velocity, all expressed
# in the robot's gravity-aligned base frame). Scaled like the genoview blocks (one shared
# std per block) then multiplied by these so box placement dominates the pick/place match.
BOX_POS_WEIGHT = 2.0
# Box orientation is weighted heavily so the CARRY search stays "sticky" to the box's current
# orientation. The carry clips genuinely hold the (near-square) box at orientations up to ~90
# apart, so a light weight lets the search hop between them and the box visibly spins; at 5.0
# the box-in-base yaw wanders only ~6 deg over a whole carry (vs ~170 deg at 1.0). Carry body
# poses are homogeneous, so this barely affects the gait match.
BOX_ROT_WEIGHT = 2.0
BOX_VEL_WEIGHT = 0.5
# PICK has its own (separate) database, so it can weight the box position AND orientation more
# than carry/place do: when you press B, the entry should be chosen mostly by where the box
# sits (and which way it faces) relative to the robot, even if the body pose matches a little
# worse -- the pose pop is inertialized away, but a bad box placement is visible. With the
# single SceneBot pick these weights select the entry frame and the box-yaw fold.
PICK_BOX_POS_WEIGHT = 5.0
PICK_BOX_ROT_WEIGHT = 5.0
BOX_INERT_HALFLIFE = 0.1  # box pose-transition (attach) inertialization half-life
