# The box task for the Unitree G1

The G1 walks to a box, picks it up, carries it and puts it down, the way of lmm-g1-box's page
([jhkim.me/lmm-g1-box](https://jhkim.me/lmm-g1-box/?physics=1&tracker=sonic_release&variant=holosoma-v3-box338x335x360-apv4-scenebox-sbplace-angle60&tab=play):
the variant `holosoma-v3-box338x335x360-apv4-scenebox-sbplace-angle60`, the Play tab, physics on, the tracker
`sonic_release`). The page's controller is ported here from its JavaScript (lmm-g1-box `web/js/`, commit `1721681`)
to Python in `mm_g1/`, and the page's motion data is copied into `data/` (`data/README.md` lists each file, its
SHA-256 and its terms). Nothing here reads lmm-g1-box.

`sonic_g1_box/run_slam.py` streams the reference to the SONIC node of
[motionmatching-g1-deploy](https://github.com/whitealex95/motionmatching-g1-deploy), on the real G1 or in that repo's
MuJoCo sim. The robot's pose comes from the map server (the Mid-360 with FAST-LIO2) and the box from `live.sam3box`
(SAM3 on the G1's D435i). The sessions are in that repo's `README_BOX_SIM.md` and `README_BOX_REAL.md`.

## Files

| file | port of (lmm-g1-box `web/js/`) | what it holds |
|---|---|---|
| `mm_g1/database.py` | `controller.js` (`loadMMData`, `loadLoco`, `MMBackend`) | the Animation Database, the Box Matching Database and the Locomotion Matching Database (`emmboth` is built at load time from `emm_loco` and `emm_carry`), exact search |
| `mm_g1/controller.py` | `controller.js` (`Controller`) | Locomotion and the box controller, inertialization, the simulation object and trajectory, root warping, the moving-target rules |
| `mm_g1/ik.py` | `ik.js` | foot locking |
| `mm_g1/sim.py` | `sim.js`, `scenario.js` | the boxes, the pick and place targets, the grasp and the release, the scripted sequence |
| `mm_g1/mathq.py` | `math.js` | vectors, quaternions, springs |
| `mm_g1/runner.py` | `phys/runner.js` | the stream for a tracked robot: 60 Hz poses resampled to 50 Hz frames, a re-plan every 10 frames with snap-xyyaw, no seed while bent down, the grip check, the box sync |
| `sonic_g1_box/run_slam.py` | `phys/engine.js` (`graspStep`), `app.js` | the bridge: the SLAM pose, the box reports, the palm force, the keys, the headless sequence |
| `sonic_g1_box/box_window.py` | | the bridge's window (UI 2): the loco driver's window and keys, plus the box task |
| `sonic_g1_box/route_follow.py` | | UI 1's trajectory, walked with `T`: a copy of the loco driver's `RouteFollow` |
| `sonic_g1_box/operator_link.py` | | UI 3's link through the map server: its keys and commands, the bridge's flags, the restart. The same file as the loco driver's |
| `sonic_g1_box/deploy_link.py` | | the node's ZMQ topics and the desktop's JSON bus, mirrored from motionmatching-g1-deploy |
| `tools/page_parity/` | | the parity check against the page's own modules, run in node |
| `assets/unitree_g1/` | | the G1 model the window draws |

## Install

The bridge runs in motionmatching-g1-deploy's conda env `mm-g1-deploy` (that repo's `INSTALL.md`). On its own it
needs Python 3.10 or newer and `pip install -r requirements.txt`. The window needs an OpenGL display (`libglfw3`).
The parity check needs node and a checkout of lmm-g1-box.

## Running it

In motionmatching-g1-deploy, the box sessions of `README_BOX_SIM.md` and `README_BOX_REAL.md` (*The session*) run
`python sonic_g1_box/run_slam.py` from this repo's root: its window (UI 2) by default, `--terminal` for letter +
Enter keys, `--web` for no window (UI 3, the map server's `/ui3` page, drives it, and takes part in the other modes
too). UI 1's *Box task* card (the map server's `box_task` op) sets a pick, a carry and a put-down that `BoxRoute` in
`run_slam.py` walks in POSE. The window has the keys of motionmatching-g1-loco's MM driver window (`]`, `P`, `O`, `Enter`, `W A S D`,
the arrows, `Shift`, `F`, `T`, `Esc`), and the box task adds `X`, a click on a reported box, `B`, `C`, `V` and `L`.
The arrows are not the page's: they turn the character (the controller's optional `face` input), and like the page's
keys they take over from a pick or a place while held. Headless, that repo's launcher:

```bash
tools/headless_box_run.sh $PWD/runs/box_pick --secs 300
tools/headless_box_run.sh $PWD/runs/box_shift --secs 300 --box-shift 0.5
```

The box must be the variant's, 0.338 x 0.335 x 0.36 m. `README_BOX_SIM.md` (*The bridge*, *Results*) and
`README_BOX_REAL.md` (*Keys in the box bridge window*, *Bridge options*) have the keys, the results and what
each default below is for.

## What differs from the page

The page simulates the robot. Here the robot is measured, a little late and with some error, and these differ
(each a default of `run_slam.py`):

- `--lookahead 60` (the page: 25). The node plays a streamed frame only once 46 frames follow it.
- The seed compares the SLAM pelvis with the reference frame of the pose's own time plus `--track-delay 0.1` s, and
  turns `--yaw-gain 0.3` of the yaw gap (the page: the frame tracked now, all of the gap). Before POSE it moves the
  controller onto the robot with all of the yaw, as the loco driver does.
- In POSE each frame's heading is sent without the seeds' turns since POSE started. The deploy node takes the stream's
  heading at the start of POSE as the robot's and turns the robot with every change of it, so a seed sent with its
  turn turned the robot as much as the reference, and a yaw gap at the start of POSE never closed: both turned on
  forever.
- In planner mode the keys go to the node's planner (the page has none), turned from the map frame into the planner's
  own, whose +x is the robot's heading when the planner started (the node's heading state on `g1_debug`).
- The box is `live.sam3box`'s report, a few times a second and 0.2 to 0.6 s late. A move is taken after
  `--sync-need 2` reports that agree, the grip check after `--grip-need 2` reports.
- The palm force aims at a fresh report of the box on the floor until the reference lifts the box, then at the box
  where the reference holds it (the page: the physical box always).
- In locomotion no seed puts the controller's simulation object inside a box's collision zone (the box's half size
  plus `collisionRadius` 0.36 m). The controller pushes the simulation object out of the zone at its next step, so
  such a seed is undone within 0.1 s. After a place the robot often stands 0.2 to 0.3 m closer to the box than the
  reference, and seeding it every 0.2 s made the reference shuffle in place until the robot kicked the box.
- A grip-check drop clears the place target. On the page the drop comes during the rise, before any place is asked
  for. Here the D435i sees the box left on the floor some 2 s after the grasp, and a place asked for meanwhile made
  the controller put down the box right after its next lift, held or not.

## Parity with the page

```bash
node tools/page_parity/js_trace.mjs ~/Projects/lmm-g1-box /tmp/js.jsonl --states /tmp/states.jsonl
python tools/page_parity/check_parity.py /tmp/js.jsonl --forced /tmp/states.jsonl
node tools/page_parity/js_match_cases.mjs ~/Projects/lmm-g1-box /tmp/cases.json
python tools/page_parity/check_parity.py /tmp/cases.json --cases
```

The forced check loads the page's whole state before each frame of the page's scripted sequence, steps once and
compares the state after it. A free run follows the page until rounding decides a near tie in the search, and from
there it takes another path of the same controller. In one test the costs were 4.1112789 and 4.1112797 at frame 484.

The scripted sequence gives the same trace with lmm-g1-box `a41bd3f` and `1721681`: there the search never jumps
while a SceneBot pick or place plays before its grasp or release, so the rules for these frames are not checked by
it. `--cases` checks `Controller.match` alone on 3,424 searches made around the SceneBot pick and place. The playing
frame is every 5th frame of each SceneBot range plus 60 random frames, and each query is a database frame's features
plus noise. The rule for the place that `1721681` added decides 420 of them, so a port of `a41bd3f` fails this check.

## Results

In the sim with the LiDAR, FAST-LIO2 and SAM3 (motionmatching-g1-deploy's `README_BOX_SIM.md` *Results* has the
tables): picking the box up and putting it down in front worked on the first try in all four runs, the box landing
0.06 to 0.20 m beyond the place target. When the box jumps 0.5 m back as the hands reach it, the regrasp is not
reliable: one run lifted it on the first regrasp, one on the second, four on the third to fifth crouch, and one not in
300 s. The regrasp bends down 0.5 to 0.9 m from the box, and the hands reach it from about 0.5 m. One run of each
scenario (pick9, shift7) ran the port of `1721681`, the others the port of `a41bd3f`. Untested on the robot.

## Credits

- G1 model: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `unitree_g1` (license in
  `assets/unitree_g1/LICENSE`).
- Motion data and method: `data/README.md`.
