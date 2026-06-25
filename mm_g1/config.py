"""Shared constants: paths, skeleton layout, and motion-matching settings.

Everything is resolved relative to this file so the project is fully relocatable --
clone the folder anywhere and the model, data and cache paths still line up.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "gmr_lafan1_g1")   # GMR-retargeted LAFAN1 .pkl clips
JUMP_DATA_DIR = os.path.join(ROOT, "data", "g1_jump")     # CAMDM walk->jump->walk .csv clips
BOX_DATA_DIR = os.path.join(ROOT, "data", "robot_object_g1")  # OmniRetarget pick/carry/place .npz
SCENE_XML = os.path.join(ROOT, "assets", "unitree_g1", "scene.xml")       # G1 only (FK + loco)
SCENE_BOX_XML = os.path.join(ROOT, "assets", "unitree_g1", "scene_box.xml")  # G1 + box (interactive)
LIB_PATH = os.path.join(ROOT, "data", "motion_lib.npz")   # built on first run, then cached

FPS = 30
DT = 1.0 / FPS

# qpos layout (36-D), shared by the dataset and MuJoCo (same joint order):
#   [0:3]  root position (x, y, z) in world metres
#   [3:7]  root orientation quaternion -- DATASET stores xyzw, MuJoCo qpos stores wxyz
#          (csv_to_qpos / mirror_qpos handle the reorder; see g1_model.py)
#   [7:36] 29 joint angles (radians)
JOINTS = slice(7, 36)

# Foot bodies used for motion-matching pose features (names from menagerie g1.xml).
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]

# --- Motion-matching: GenoView (Holden "Simple Motion Matching") heuristics ----------
# All math + hyperparameters mirror ../GenoViewPython-MotionMatching/genoview_g1.py.
HORIZONS = [10, 20, 30]        # future trajectory taps (frames) ~0.33 / 0.67 / 1.0 s @30fps
TRAJ_HORIZONS = HORIZONS       # (alias kept for any external reference)
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

# GenoView-matched absolute [start:stop] frame windows (30 fps; genoview's 60 fps indices
# halved). These trim the T-pose/calibration off walk & run and -- crucially -- isolate the
# ~5 s stumble EVENT out of the otherwise-ordinary pushAndStumble clip. Keeping the whole
# pushAndStumble clip instead leaves thousands of standing/walking frames that the matcher
# settles into when you STOP, producing a different idle than genoview.
CLIP_TRIM = {
    "walk1_subject5":            (80, 7759),   # genoview 160-15518 @60fps
    "run1_subject5":             (86, 7068),   # genoview 172-14136 @60fps
    "pushAndStumble1_subject5":  (198, 353),   # genoview 397-706  @60fps  (stumble event only)
}

# Bump when the library build (clips, trims, mirror, labels) changes incompatibly, so a
# stale data/motion_lib.npz cache is rebuilt automatically. v2: GenoView-matched trims.
# v3: robot-object pick/carry/place skill + box features.
LIB_VERSION = 3

# GenoView trims the last HORIZONS[-1] frames of each clip from the SEARCH only
# (cKDTree(X[rs:re-30])): the tail still plays out, but a match never lands there, so a
# full future trajectory always exists and the playhead can't run off the clip end.
SEARCH_TAIL = HORIZONS[-1]   # frames excluded from each clip's KD-tree (1.0 s @30fps)

# The locomotion library: GMR-retargeted LAFAN1 clips (subject5) -- walk, run, and
# push-and-stumble. Each clip is added twice (normal + L/R MIRRORED, GenoView-style) for
# symmetric left/right coverage. A desired-velocity command then steers motion matching
# smoothly between standing-walk, run, and the stumble-recovery variety.
CLIPS = ["walk1_subject5", "run1_subject5", "pushAndStumble1_subject5"]
MIRROR = True                  # append a left/right-mirrored copy of every clip

# Desired locomotion speed (m/s) fed to the trajectory springs. Full stick = MAX_SPEED;
# holding Shift scales it to a walk (GenoView's 0.4 scale).
MAX_SPEED = 2.5
WALK_SCALE = 0.4

# Full-stick command speed while CARRYING the box. The carry clips are near-stationary -- the
# root translates at ~0.5 m/s on average (p95 ~0.75, max ~0.97), so commanding the locomotion
# speed here would just ask for motion the carry data cannot supply. Capping the command at
# the data's ~p95 lets WASD nudge the carry as fast as the clips actually move, no faster.
CARRY_MAX_SPEED = 0.75

# --- Jump skill (triggered with J) ----------------------------------------------
# CAMDM walk->jump->walk clips (in JUMP_DATA_DIR, 30 fps CSVs, same 36-D layout). They
# are appended to the library and phase-labeled; a jump is ENTERED only from its run-up
# (the `ready` phase) and ridden through landing, never matched into during locomotion.
JUMP_CLIPS = ["walk_jump_walk", "walk_jump_walk2", "walk_jump_stop"]
JUMP_FOOT_THR = 0.13     # m: both feet above this == airborne (flight detection)

# Five phases carved around each detected flight (frame counts @30fps). A jump is entered
# only in `ready` (the run-up before push-off) and exited only after `after` (recovery).
JUMP_PHASES = ["walk", "ready", "takeoff", "flight", "touchdown", "after"]
PHASE_READY = 12         # run-up before push-off -- the only place a jump can be entered
PHASE_TAKEOFF = 10       # ground push-off / loading just before lift-off
PHASE_TOUCHDOWN = 6      # landing impact, just after the feet hit
PHASE_AFTER = 18         # landing absorption / recovery walk -- the only place to exit

# --- Box manipulation skill (pick / carry / place; triggered with B) -------------
# OmniRetarget robot-object .npz clips (in BOX_DATA_DIR, 30 fps). Each clip is one full
# pick -> carry -> place sequence: the G1 lifts a large box off the floor, holds it, and
# sets it back down. Layout per frame is MuJoCo qpos (43-D): robot [0:36] (same 36-D as the
# locomotion data) then box freejoint [36:39] pos + [39:43] quat (wxyz).
#
# We segment every clip into the three phases and drive them from a small state machine:
#   LOCOMOTION --B (near box)--> PICK (ride) --> CARRY (search) --B--> PLACE (ride) --> LOCOMOTION
# PICK and PLACE are ridden through like the jump (no search mid-skill); CARRY is searched
# the same way as locomotion but only among CARRY frames, with box features added.
BOX_CLIPS = "all"        # "all" -> every .npz in BOX_DATA_DIR, or an explicit list of stems

# Per-frame skill codes (lib["skill"]). 0 keeps locomotion exactly as before; any non-zero
# code keeps that frame out of the locomotion search/normalization. JUMP is the legacy J
# skill (kept distinct from the box codes so both can coexist in one library).
SKILL_LOCO, SKILL_PICK, SKILL_CARRY, SKILL_PLACE, SKILL_JUMP = 0, 1, 2, 3, 4

# Phase segmentation thresholds (box height relative to its resting height on the floor).
BOX_REST_FRAMES = 5      # frames averaged at clip start to estimate the resting box height
BOX_CARRY_FRAC = 0.6     # box above rest + this*(peak-rest) == the CARRY plateau
# "Held" (box attached to the robot, moving with it): box lifted clear of the floor OR
# moving. Drives when the box rides the robot vs. rests in the world (pick contact / place
# release). Box speed is m/s.
BOX_HOLD_DZ = 0.10       # m above resting height to count as lifted
BOX_HOLD_SPEED = 0.05    # m/s box speed to count as being handled

# Entry windows: a PICK/PLACE is entered only in the first few frames of its phase (the
# reach-down / set-down approach), nearest-neighbour matched to the live pose + box pose,
# then ridden to the phase end -- mirrors the jump's `ready` entry.
PICK_ENTRY = 8           # candidate entry frames at the start of each PICK phase
PLACE_ENTRY = 8          # candidate entry frames at the start of each PLACE phase

# B only picks up a box you are standing next to (planar root<->box distance, metres).
PICK_RADIUS = 0.9

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
# PICK has its own (separate) database, so it can weight the box position more than carry/place
# do: when you press B, the entry should be chosen mostly by which pick clip's box sits where
# YOUR box actually is relative to the robot (correct reach), even if its body pose matches a
# little worse -- the pose pop is inertialized away, but a bad box position is visible.
PICK_BOX_POS_WEIGHT = 5.0
BOX_INERT_HALFLIFE = 0.1  # box pose-transition (attach) inertialization half-life

# Where the box spawns, expressed in the robot's start frame (so it is always a reachable
# distance in front of wherever the character begins / resets), plus its resting height. The
# resting orientation is taken from the data so the pick entry lines up (see controller).
BOX_SPAWN_FWD = 1.6      # metres in front of the robot's start facing
BOX_SPAWN_LAT = 0.0      # metres to the robot's left (+) / right (-)
BOX_REST_Z = 0.19
