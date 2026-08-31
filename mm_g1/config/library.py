"""What the motion library contains and how it is baked: clips, trims, folds,
phase segmentation, and the box geometry. Changing anything here changes the
data/motion_lib.npz cache -- bump LIB_VERSION when the change is incompatible.
"""

FPS = 30
DT = 1.0 / FPS

# qpos layout (36-D), shared by the dataset and MuJoCo (same joint order):
#   [0:3]  root position (x, y, z) in world metres
#   [3:7]  root orientation quaternion -- DATASET stores xyzw, MuJoCo qpos stores wxyz
#          (csv_to_qpos / mirror_qpos handle the reorder; see g1_model.py)
#   [7:36] 29 joint angles (radians)

# Foot bodies used for motion-matching pose features (names from menagerie g1.xml).
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]

HORIZONS = [10, 20, 30]        # future trajectory taps (frames) ~0.33 / 0.67 / 1.0 s @30fps

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
# v3: robot-object pick/carry/place skill + box features. v4: box orientation N-fold augmentation.
# v5: jump skill removed. v6: 7 low-quality box clips excluded (BOX_CLIPS_EXCLUDE).
# v7: single SceneBot pick/drop (clip 11 half-speed + reversed), baked contact labels,
#     OmniRetarget clips reduced to carry-only, SceneBot 0.3x0.2x0.3 box.
# v8: per-frame label renamed skill -> phase (lib key + Phase enum).
LIB_VERSION = 8

# The locomotion library: GMR-retargeted LAFAN1 clips (subject5) -- walk, run, and
# push-and-stumble. Each clip is added twice (normal + L/R MIRRORED, GenoView-style) for
# symmetric left/right coverage. A desired-velocity command then steers motion matching
# smoothly between standing-walk, run, and the stumble-recovery variety.
CLIPS = ["walk1_subject5", "run1_subject5", "pushAndStumble1_subject5"]
MIRROR = True                  # append a left/right-mirrored copy of every clip

# --- Box manipulation skill data (pick / carry / place) --------------------------
# OmniRetarget robot-object .npz clips (in BOX_DATA_DIR, 30 fps). Each clip is one full
# pick -> carry -> place sequence: the G1 lifts a large box off the floor, holds it, and
# sets it back down. Layout per frame is MuJoCo qpos (43-D): robot [0:36] (same 36-D as the
# locomotion data) then box freejoint [36:39] pos + [39:43] quat (wxyz).
BOX_CLIPS = "all"        # "all" -> every .npz in BOX_DATA_DIR, or an explicit list of stems

# Clips rejected after visual review of the kinematic playback (low retarget quality).
BOX_CLIPS_EXCLUDE = [
    "sub12_largebox_077_original_mujoco",
    "sub16_largebox_047_original_mujoco",
    "sub16_largebox_048_original_mujoco",
    "sub3_largebox_003_original_mujoco",
    "sub8_largebox_006_original_mujoco",
    "sub8_largebox_045_original_mujoco",
    "sub8_largebox_047_original_mujoco",
]

# Rotational augmentation of the box clips. Each pick/carry/place clip is replicated
# BOX_ROT_FOLDS times with the box's ORIENTATION yawed about its own centre by whole turns / N
# (0, 90, 180, 270 deg at N=4). Only the box quaternion is rotated -- its centre position and
# the entire robot motion are identical across folds -- so the pick/carry/place search covers
# the (near-square) box at any facing and you can lift it whichever way it is turned. Valid
# because the box has ~4-fold rotational symmetry about its vertical axis; 1 == no augmentation.
BOX_ROT_FOLDS = 4

# --- The single SceneBot pick / drop (this branch) -------------------------------
# The pick and place skills come from ONE motion: the SceneBot web demo's squat pickup
# (clip 11 of assets/scenebot/clips.bin, frames 0..120 at 50 Hz). The demo plays it at
# half speed forward for the pickup and at full speed backward for the put-down; both
# playbacks are baked as-played into the library at FPS, with the demo's per-frame
# contact labels (mm_g1/scenebot_pick.py). The OmniRetarget clips then contribute ONLY
# their carry frames (their own pick/place phases are marked Phase.DISABLED).
SCENEBOT_PICK = True
SCENEBOT_CLIP = 11
SCENEBOT_FRAMES = (0, 120)
SCENEBOT_FPS = 50
SCENEBOT_PICK_SPEED = 0.5      # the demo's pickupForwardStepScale
SCENEBOT_PLACE_SPEED = 1.0     # reverse playback runs at full speed in the demo
# The SceneBot box (0.3 x 0.2 x 0.3 m) is only 2-fold rotationally symmetric about
# vertical, so the baked pick/drop is replicated at 0 and 180 deg of box yaw only.
# (The approach heuristic aligns the stance to the live box yaw, so 2 folds suffice.)
SCENEBOT_ROT_FOLDS = 2
BOX_HALF = (0.15, 0.10, 0.15)  # SceneBot free_box half extents
BOX_REST_Z = BOX_HALF[2]       # the SceneBot box rests on the floor at its half height

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
# then ridden to the phase end.
PICK_ENTRY = 8           # candidate entry frames at the start of each PICK phase
PLACE_ENTRY = 8          # candidate entry frames at the start of each PLACE phase
