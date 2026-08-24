# Interactive Motion Matching for the Unitree G1 — with box pick / carry / place

Steer a **Unitree G1** humanoid around a MuJoCo scene in real time with the keyboard, and
**pick up, carry, and set down a box** on command. Hold **WASD** and a
[motion-matching](https://www.gdcvault.com/play/1023280/Motion-Matching-and-The-Road)
search stitches GMR-retargeted LAFAN1 **walk**, **run** and **push-and-stumble** clips
into one continuous, responsive gait; press **B** next to the box and the controller
switches to a small **pick → carry → place** state machine driven by
[OmniRetarget](https://github.com/) robot-object clips — no neural network, no training,
just nearest-neighbour search over per-skill feature databases.

This repo is **fully self-contained**: the G1 model, all motion data, and the code live in
this one folder. Clone it, run `setup.sh`, and go.

```
W / A / S / D    move (relative to the camera)
Shift (hold)     run instead of walk
B                box action: pick up when near the box, set down while carrying
Space            reset to the start pose
T                toggle the command trajectory gizmo
left-drag        orbit camera
right-drag       pan camera
scroll           zoom
Esc              quit
```

A box spawns a short distance in front of the character. **Press B from anywhere**: the
controller plans a walking route to the pick stance in front of the box (green line),
walks it, settles, and rides the pick clip; the box then rides along in its hands. You are
now **carrying** — move around (the box follows), then press **B** again to play the
put-down and set the box back on the floor. B during the walk-over cancels it.

A red **command gizmo** (à la GenoView's `DrawTrajectory`) is drawn on the ground: a
sphere at each predicted future position with a short stick pointing in the predicted
facing direction. It shows exactly the trajectory the matcher is being asked to follow —
press **T** to toggle it.

## Box manipulation (pick / carry / place)

On this branch (`scenebot_g1_box_single`) the pick and the drop come from **one single
motion**: the SceneBot web demo's squat pickup (clip 11 of the vendored motion graph,
frames 0..120 at 50 Hz). The demo plays it at **half speed forward** for the pickup and
the **same frames backward at full speed** for the put-down, and both playbacks are baked
into the library exactly as played (`mm_g1/scenebot_pick.py`), resampled to 30 Hz,
**together with the demo's own per-frame contact labels** (feet + wrists, `lib['contact']`).
The clip has no box, so a box trajectory for the SceneBot free box (0.3 × 0.2 × 0.3 m) is
synthesized: it rests where the clip's hands close (0.29 m ahead of the stance) and rides
the wrist midpoint from that frame on. The [OmniRetarget](https://github.com/) robot-object
clips contribute **only their carry frames** (their own pick/place phases are disabled);
`mm_g1/boxes.py` still segments them and marks where the box is **attached**.

A state machine with a shelf-style approach drives it (`mm_g1/controller.py`):

```
LOCOMOTION --B--> MOVE-TO-PICK (walk the planned route, settle) --> PICK (ride)
    --> CARRY (search) --B--> PLACE (ride) --> LOCOMOTION
```

**MOVE-TO-PICK** (approach heuristics, ported from `motionmatching-g1-shelf`): B inverts
the recorded stance-to-box offset at the LIVE box pose (the box's 2-fold symmetry gives
two stance candidates; the nearer way-in point wins), plans a polyline route — straight to
a way-in point 0.6 m behind the stance, a rounded corner, then in along the stance heading
— and synthesizes the walk command from a look-ahead point on it, with the matcher's
future trajectory taps read straight off the route. On the final leg the root is pinned to
the rail (cross-track and yaw errors decay with a 1 s half-life); near the stance it
servos to a stop, and the pick entry fires once the root has settled there. Unlike the
shelf demo (which cuts into its clip at the stance crossing, still walking), the entry
waits for a settled stop: the SceneBot tracking policy cannot follow an instant reference
stop, and the demo's own sequence also settles before the squat.

- **PICK** and **PLACE** are *ridden*: entered from the start of the
  phase by a nearest-neighbour match of the live pose **+ box pose**, then played to the
  phase end with no mid-skill search.
- **CARRY** is searched every `SEARCH_TIME` like locomotion, but only among `carry` frames,
  with the box pose added to the query.
- The **only** database transitions ever made are those in the chain above (so e.g. you can
  never match from locomotion straight into a carry, or from a pick into a place).

Each searchable database has its **own feature space** (`mm_g1/features.py`), all expressed
in the smoothed sim-root (gravity-aligned base) frame:

| database | dims | contents                                                       |
|----------|------|----------------------------------------------------------------|
| `loco`   | 27   | pose (15) + future trajectory (12)        — unchanged genoview  |
| `carry`  | 36   | pose + future trajectory + **box** pos/orient/vel (9)          |
| `pick`/`place` (`pp`) | 24 | pose + **box** pos/orient/vel — **no** future trajectory |

The box block is `[ position (3) · orientation as scaled-angle-axis (3) · linear velocity
(3) ]` in the base frame. While the box is held it is reconstructed each frame as
`root ∘ box-in-base` (exactly like the pelvis), so it tracks the character; a short
inertialization offset captured at grab time hides the pop from its resting spot.

## Quick start

```bash
git clone <this-repo> motionmatching-g1
cd motionmatchin-g1
./setup.sh                      # makes .venv, installs deps, builds the cache
source .venv/bin/activate
python run.py                   # opens the window — WASD to move
```

The first launch builds a feature cache (`data/motion_lib.npz`, ~1 s); later launches
start instantly. The viewer needs a display — run it on a desktop or an X-forwarded
session with `MUJOCO_GL=glfw` (the default).

Already have a MuJoCo Python environment? Skip `setup.sh`:

```bash
pip install -r requirements.txt
python run.py
```

## How it works

Each frame the loop does four things:

1. **Read the keys → a desired trajectory.** Held WASD become a desired
   `(speed, heading)` relative to the camera. `predict_trajectory` slews the heading
   toward the input at a fixed turn rate and integrates forward to a short predicted
   path (`mm_g1/commands.py`).
2. **Build a query and search.** The query is `[ predicted trajectory | the current
   frame's pose features ]`, standardized the same way as the database. A KD-tree
   returns the nearest library frame (`mm_g1/controller.py`).
3. **Continue or jump, with hysteresis.** We only switch to the nearest neighbour when
   it is clearly better (by `jump_margin`) than continuing the current clip — this keeps
   the character on long continuous fragments, so the motion stays smooth.
4. **Stitch and blend.** The chosen frame is placed into the world by a planar (SE2)
   alignment so the root path is C0-continuous; pose pops at a switch are cross-faded
   over ~0.4 s (`mm_g1/kinematics.py`).

The feature vector (27-D, `mm_g1/features.py`) is computed entirely in each frame's
root-local frame so matching is heading-invariant:

| block      | dims | contents                                                |
|------------|------|---------------------------------------------------------|
| trajectory | 12   | future root offset + facing at +10/+20/+30 frames       |
| pose       | 15   | local foot positions (2×3), foot velocities (2×3), root velocity (3) |

Searching every `MM_SEARCH_INTERVAL` frames (default 15, ~0.5 s) rather than every frame
reduces pops; commanding a higher speed (Shift) pulls the match into the **run** clip,
a lower speed back into the **walk** clip.

## Layout

```
motionmatching-g1-box/
├── run.py                       # entry point: python run.py
├── setup.sh                     # venv + install + build cache (self-contained)
├── requirements.txt
├── mm_g1/
│   ├── config.py                # paths, FPS, joint layout, feature + skill settings
│   ├── g1_model.py              # qpos conversion, quaternion yaw, FK for the feet, mirror
│   ├── data.py                  # build / load + cache the loco + box library
│   ├── boxes.py                 # pick/carry/place segmentation + entry indexing
│   ├── features.py              # per-skill feature DBs (loco 27 / carry 36 / pick·place 24)
│   ├── springs.py               # critically-damped trajectory + inertialization springs
│   ├── controller.py            # real-time matcher + pick/carry/place state machine
│   └── viewer.py                # GLFW + MuJoCo window, held-key input, follow-camera, box
├── sonic_g1_box/                # the motion-matched motion tracked by SONIC in physics
├── scenebot_g1_box/             # the box pickup tracked by the SceneBot demo's policy
├── tools/sonic_tracking/        # SONIC ONNX policy port + G1 deploy constants
├── tools/scenebot_tracking/     # SceneBot policy + motion-graph port (from their web demo)
├── assets/unitree_g1/           # MuJoCo G1 model (g1.xml, scene.xml, scene_box.xml, meshes)
├── assets/sonic/                # SONIC policy weights + NVIDIA 29-DoF G1 scene
├── assets/scenebot/             # SceneBot policy, clips, motion graph, flat-hand G1 (vendored)
├── assets/largebox/             # the box mesh (largebox.obj)
├── data/gmr_lafan1_g1/          # GMR-retargeted LAFAN1 clips (walk / run / pushAndStumble, .pkl)
└── data/robot_object_g1/        # OmniRetarget robot-object pick/carry/place clips (.npz)
```

## Tuning

Edit `mm_g1/config.py`:

- `WALK_SPEED` / `RUN_SPEED` — commanded speeds (m/s) for walk and Shift-run.
- `TURN_RATE` — how fast the predicted heading chases the input direction.
- `MM_SEARCH_INTERVAL` — frames between searches (lower = more reactive, more pops).
- `CLIPS` — which clips form the library (drop extra GMR `.pkl` clips into
  `data/gmr_lafan1_g1/` and list them here; delete `data/motion_lib.npz` to rebuild).
- `CLIP_TRIM` / `DEFAULT_TRIM` — per-clip `(head, tail)` frames cropped to remove the
  LAFAN1 T-pose lead-in/out. These mirror GenoView's hand-picked `start:stop` indices
  (its 60 fps starts of walk=160 / run=172 → our 30 fps 80 / 86); add an entry per new clip.
- `SEARCH_TAIL` — frames at each clip's end excluded from the *search only* (GenoView's
  `cKDTree(X[rs:re-60])`): the tail still plays but can't be matched into, so the
  character never runs off the end of a clip.

Box-skill knobs (also in `config.py`):

- `BOX_CLIPS` — which robot-object clips to load (`"all"` or an explicit list of stems).
- `PICK_RADIUS` — how close the root must be to the box for **B** to pick it up.
- `BOX_SPAWN_FWD` / `BOX_SPAWN_LAT` — where the box spawns, in the robot's start frame.
- `BOX_CARRY_FRAC`, `BOX_HOLD_DZ`, `BOX_HOLD_SPEED` — phase-segmentation thresholds
  (`mm_g1/boxes.py`): how high the box must rise to count as *carry* / be *attached*.
- `BOX_POS_WEIGHT` / `BOX_ROT_WEIGHT` / `BOX_VEL_WEIGHT` — how much the box blocks weigh in
  the pick/place/carry search vs. the body pose.
- `BOX_INERT_HALFLIFE` — how quickly the box settles into the hands at grab time.

> **Note on the carry data.** The OmniRetarget carry clips are essentially in-place (the
> robot holds the box and barely translates), so while *carrying* the character mostly
> stands/shuffles — WASD has limited effect until you set the box down. This follows the
> spec faithfully (carry searches only `carry` frames); swap in walking-while-carrying data
> and the same machinery would steer it.

## Credits

- **G1 model** — [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
  `unitree_g1` (license in `assets/unitree_g1/LICENSE`).
- **Motion data** — [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
  (Ubisoft La Forge), retargeted to the G1 with
  [GMR (General Motion Retargeting)](https://github.com/YanjieZe/GMR). See
  `data/gmr_lafan1_g1/README.md` for the clip list and pickle format.
- **Approach** — adapted from a MuJoCo G1 motion-matching / motion-graph project and the
  real-time, keyboard-driven control of GenoView's motion-matching demo.
