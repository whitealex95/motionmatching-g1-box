# The box task for the Unitree G1

The G1 walks to a box, picks it up, carries it and puts it down, the way of lmm-g1-box's page
([jhkim.me/lmm-g1-box](https://jhkim.me/lmm-g1-box/?physics=1&tracker=sonic_release&variant=holosoma-v3-box338x335x360-apv5-good2-sbplace-angle60&tab=play):
the variant `holosoma-v3-box338x335x360-apv5-good2-sbplace-angle60`, the Play tab, physics on, the tracker
`sonic_release`, the page's defaults of its tag `v4.2.3`, `9c45ac7`). The page's controller and its pick motion (the pick
editor's whole-body IK, which solves the pick and the put-down again for the height under the box) are ported here
from its JavaScript (lmm-g1-box `web/js/`) to Python in `mm_g1/`, and the page's motion data is copied into `data/`
(`data/README.md` lists each file, its SHA-256 and its terms). Nothing here reads lmm-g1-box.

`sonic_g1_box/mm_driver.py` streams the reference to the SONIC node of
[motionmatching-g1-deploy](https://github.com/whitealex95/motionmatching-g1-deploy), on the real G1 or in that repo's
MuJoCo sim. The robot's pose comes from the map server (the Mid-360 with FAST-LIO2) and the box from `live.sam3box`
(SAM3 on the G1's D435i). The sessions are in that repo's `README_BOX_SIM.md` and `README_BOX_REAL.md`.

## Files

| file | port of (lmm-g1-box `web/js/`) | what it holds |
|---|---|---|
| `mm_g1/database.py` | `mm/controller.js` (`loadMMData`, `loadLoco`, `MMBackend`) | the Animation Database, the Box Matching Database and the Locomotion Matching Database (`emmboth` is built at load time from `emm_loco` and `emm_carry`), exact search, the tables of the walks that end in a SceneBot clip and of the frame each bend rises at |
| `mm_g1/controller.py` | `mm/controller.js` (`Controller`) | Locomotion and the box controller, inertialization, the simulation object and trajectory, root warping, the moving-target rules, the walk that plays on into its SceneBot clip, the cancel of a bend |
| `mm_g1/ik.py` | `mm/ik.js` | foot locking |
| `mm_g1/pickmotion.py` | `mm/pickmotion.js` | the pick motion: the pick and the put-down solved again for the height under the box, the box size and the Pick motion tab's edits (the put-down's own, when it has one), put into the Animation Database |
| `mm_g1/pickedit.py`, `mm_g1/bodyik.py`, `mm_g1/mirror.py` | `edit/pickedit.js`, `edit/bodyik.js`, `edit/mirror.js` | the pick editor's solve: the whole-body IK on the simulated G1 of `data/physics/` (MuJoCo), the mirrored clips |
| `mm_g1/sim.py` | `mm/sim.js`, `mm/scenario.js` | the boxes, the pick and place targets, the grasp and the release, the scripted sequence |
| `mm_g1/mathq.py` | `mm/math.js` | vectors, quaternions, springs |
| `mm_g1/runner.py` | `phys/runner.js` | the stream for a tracked robot: 60 Hz poses resampled to 50 Hz frames, a re-plan every 10 frames with snap-xyyaw, no seed while bent down, the grip check, the box sync |
| `sonic_g1_box/mm_driver.py` | `phys/engine.js` (`graspStep`), `pages/index.js` | the driver: the SLAM pose, the box reports, the palm force, the keys, the headless sequence, the height under the box |
| `sonic_g1_box/pick_worker.py` | | the pick motion's solves on a thread of their own (0.6 to 1.7 s each), and the Pick motion card's numbers and clip |
| `sonic_g1_box/box_window.py` | `pages/index.js` (`bindUI`, the clicks) | the driver's window (UI 2): the loco driver's window and keys, plus the box task |
| `sonic_g1_box/db_page.py`, `web/` | `motions.html`, `pages/motions.js`, `ui/states.js` | the database page (http 8790): the controller's state, the clip the robot plays and the last matches, live, above the page's database page. `web/` is laid out as the page's (`js/mm/`, `js/ui/`, `js/pages/`, `css/`) and holds byte copies of the page's modules and three.js, and `js/pages/db.js`, `index.html`, `css/db.css` and `js/mm/variants.js` made from the page's |
| `sonic_g1_box/route_follow.py` | | UI 1's trajectory, walked with `T`: a copy of the loco driver's `RouteFollow` |
| `sonic_g1_box/operator_link.py` | | UI 3's link through the map server: its keys and commands, the driver's flags, the restart. The same file as the loco driver's |
| `sonic_g1_box/deploy_link.py` | | the node's ZMQ topics and the desktop's JSON bus, mirrored from motionmatching-g1-deploy |
| `tools/page_parity/` | | the parity check against the page's own modules, run in node |
| `assets/unitree_g1/` | | the G1 model the window draws |

## Install

The driver runs in motionmatching-g1-deploy's conda env `mm-g1-deploy` (that repo's `INSTALL.md`). On its own it
needs Python 3.10 or newer and `pip install -r requirements.txt`. The window needs an OpenGL display (`libglfw3`).
The parity check needs node and a checkout of lmm-g1-box.

## Running it

In motionmatching-g1-deploy, the box sessions of `README_BOX_SIM.md` and `README_BOX_REAL.md` (*The session*) run
`python -m sonic_g1_box.mm_driver` from this repo's root: its window (UI 2) by default, `--terminal` for letter +
Enter keys, `--no-window` for none (UI 3, the map server's `/ui3` page, which runs anyway, drives it, and takes part in the other modes
too). UI 1's *Box task* card (the map server's `box_task` op) sets a pick, a carry and a put-down that `BoxRoute` in
`mm_driver.py` walks in POSE. The window has the keys of motionmatching-g1-loco's MM driver window (`]`, `P`, `O`, `Enter`, `W A S D`,
the arrows, `Shift`, `F`, `T`, `Esc`), and the box task adds `X`, `B`, `C`, `V`, `L` and the page's clicks: a click on
a reported box picks it up, a click on the floor puts the held box down there, and with empty hands the robot walks
there (full stick down to 1 m, done within 0.25 m, a key takes over). UI 3's 3-D view takes the same clicks, and its
put-down marker can be dragged. The driver also serves the database page, `http://localhost:8790/` (`--db-port`, 0
for none, not headless): the controller's state on the page's state chart, the clip the robot plays on its graph
(*Play it below* plays it), each query's feature weights and the cost share of its last match, and below them the
page's list of every motion of the databases.
The arrows are not the page's: they turn the character (the controller's optional `face` input), and like the page's
keys they take over from a pick or a place while held. Headless, that repo's launcher:

```bash
tools/headless_box_run.sh $PWD/runs/box_pick --secs 300
tools/headless_box_run.sh $PWD/runs/box_shift --secs 300 --box-shift 0.5
```

The box must be the variant's, 0.338 x 0.335 x 0.36 m. `README_BOX_SIM.md` (*The driver*, *Results*) and
`README_BOX_REAL.md` (*Keys in the box driver window*, *Driver options*) have the keys, the results and what
each default below is for.

## The page's settings

The settings of the page's Play panel are flags of `mm_driver.py`, each with the page's default. Where the page takes
the default from the data variant's `config.json`, the driver takes the same value. UI 3's *Settings* tab (the deploy
repo, `README_BOX_SIM.md`, *UI 3*) sets them while the driver runs, and a restart from UI 3 keeps them.

| page setting | flag | default |
|---|---|---|
| Anchoring | `--yaw-gain` (snap-xyyaw: 1, snap-xy: 0), `--hold` | 1, `low` |
| Grasp heuristic (and its nine presets) | `--palm-force`, `--force-at` (the force on the palms or on the wrists), `--palm-target`, `--shoulder-squeeze`, `--wrist-squeeze`, `--shoulder-open` | 40 N, `palm`, `box`, 0, 0, 0.4 rad |
| Pick and place speed, slowed down | `--pick-speed`, `--pick-slow` | 1, `clip` |
| Root warping, leave when moved | `--warp`, `--leave-moved` | `pickstep`, 0.5 m |
| Box search, rising, transition cost | `--box-entry`, `--rise`, `--transition-cost` | `walk`, `finish`, 0 |
| The end of a walk | `--walk-on` (`step`: from its last step a walk plays on into the SceneBot clip with no search. `start`: from the frame where that clip starts in it. `off`: the search goes on to the end of the walk) | `step` |
| A bend ends | `--bend-out` (`cancel`: only by its grasp or release, or by a cancel, which rises at the frame of the other SceneBot clip with the same joints, with no search. `leave`: back to Locomotion while upright, a cancel only while bent, at the frame the search finds) | `cancel` |
| Upright tilt, box database | `--upright-tilt`, `--box-db` | 30 deg, `same` |
| Foot locking, a foot still in the air when its contact starts | `--foot-lock`, `--foot-land` (m above the playing frame's foot, `off`: locked at once) | `on`, 0.02 m |
| Walk to, box farther than in the data | `--loco-target` (`arrive`: the stance, arriving there in 1 s; `stance`; `centre`), `--box-ahead` | `arrive`, 0 |
| Turning in Locomotion: turn speed limit, rotation halflife | `--turn-rate` (deg/s the simulation object of Locomotion turns at most, `0`: no limit), `--rot-halflife` (s, the halflife of the spring that turns it) | 60 deg/s (no limit until v4.2.1), 0.27 s |
| Matching weights | `--weights-loco`, `--weights-arrive`, `--weights-pick`, `--weights-place` (pick and place: 7 values, or 9 with a weight for x and one for y of the box position and of the box direction) | the variant's; Arriving: trajectory positions 1, directions 1.5, velocity 0 |
| Transitions: by cost | `--box-enter-pick`, `--box-enter-place`, `--box-leave-pick`, `--box-leave-place` | 1.5, 1.5, 2.0, 2.0 (2.5, 2.5, 4.0, 4.0 until v4.0.1) |
| Transitions: side margin, only within | `--side-margin-pick`, `--side-margin-place`, `--take-over` | 0.25, 0.25, 1 m |
| Transitions: a box that leaves the hands while lifting | `--retry-lift` | `on` |
| Grasp: shoulder open from | `--shoulder-open-from` (`take`: the take-over, `step`: the last step) | `take` |
| Transitions: by distance, for a pick and for a put-down (0: by the cost only) | `--force-engage-radius`, `--place-engage-radius` | 0.6 m (1.1 m until v4.0.1), the variant's 0.4 m |
| The distance rules' constants of the page's controller, not on its panel | `--engage-radius`, `--reapproach-time` | 3 m, 1 s |
| Pick motion: adjust to the height under the box, to the box size | `--pick-height`, `--pick-size` | `on`, `off` |
| Pick motion: the box solved for, the box at rest, the palms, the grip | `--pick-box`, `--pick-forward`, `--pick-left`, `--pick-up`, `--pick-yaw`, `--pick-palms`, `--pick-grip-keep` | `data`, 0, 0, 0, 0, `0,0,0,0`, `off` |
| Pick motion: the put down has its own box at rest and palms | `--place-edit` (`pick`: the pick's, or `own`), `--place-edit-forward`, `--place-edit-left`, `--place-edit-up`, `--place-edit-yaw`, `--place-edit-palms`, `--place-edit-grip-keep` | `pick`, 0, 0, 0, 0, `0,0,0,0`, `off` |

The page's Transitions tab (lmm-g1-box v3.6.1) has one threshold for a pick and one for a place, so `--box-enter`,
`--box-leave` and `--side-margin` of the driver before are each two flags now. The page's Pick motion tab has two
more settings that are not the page's controller: the table height, below, and the box solved for, which the page
takes from its Box tab.

Since lmm-g1-box v4.1 the task has two states before a grasp and two before a release: *Stepping to the box* and
*Bending down* (Picking up before), *Stepping to the marker* and *Lowering the box* (Putting down before). The step
is the walk that ends in the SceneBot clip, the bend is that clip. With `--walk-on step` the search no longer jumps
back into a walk next to the box, and with `--bend-out cancel` a bend that loses its target (the box moved, the
target cleared, a walk key, the cost above Leave) plays the other SceneBot clip from the frame with the same
joints: a cancelled pick rises as a put-down rises, empty-handed, and a cancelled put-down rises as a pick, with
the box. While a bend plays, both SceneBot clips are solved for that bend's height and manual edit, so the palms
stay where they are at the cancel. `--walk-on off --bend-out leave --foot-land off` are the rules before.

The seed's own rules, none of them the page's, are switches on the same *Anchoring* card of UI 3, in the order a
re-plan checks them, each on by default. UI 3 highlights the page's two (`--hold`, `--yaw-gain`), has a button that
puts all of them back to their defaults, and shows each re-plan's outcome (snapped, or the rule that skipped the
snap) and how often each rule skipped one (`seed_rules` in the driver's state).

| rule | flag (off) | default |
|---|---|---|
| no seed while stopped (O) | `--skip-stopped off` | on |
| no seed without a SLAM pose from the last 0.5 s | `--skip-stale off`: the newest pose, however old | on |
| in POSE, no seed from a pose that jumped in 0.2 s | `--max-seed-jump 0` | 0.5 m |
| in Locomotion, no seed into a box's collision zone | `--keep-out off` | on |
| before POSE, all of the yaw against the frame of the pose's time | `--pre-pose-full off`: as in POSE | on |
| in POSE, compare with the frame this much before the pose's time | `--track-delay 0` | 0.1 s |
| in POSE, the heading sent leaves out the seeds' turns | `--strip-seed-turns off`: the robot turns with each seed | on |

The squeezes turn the shoulder roll and wrist yaw targets inward through the node's `arm_tau` (q_target += tau / kp),
so `--shoulder-kp` and `--wrist-kp` must be the node's kps: 40 and 20 on the robot (`--debug-arm-pd-gains`). Not
ported: the tracker choice (the node runs SONIC), the other anchoring modes, the pick and place check, the box mass
and the number of sides.

## The height under the box

The page knows its tables. The driver has the box reports and the map:

- The pick: a box reported with its bottom `--table-min` (0.1 m) or higher stands on something that high, else on the
  floor. live.sam3box puts every bottom below 0.08 m on the floor, so a box on the floor never comes near the
  threshold. The height is rounded to 1 cm, and a change below 3 cm is not taken (the solve is 0.6 to 1.7 s).
- The put-down: the median height of the global map within 0.1 m of the marker, when half of those cells or more are
  occupied and it is from `--table-min` to 1.2 m, else the floor. The map server must map (its `--no-integrate` off,
  the box sessions' default since 2026-11-16).

With `--pick-height on`, the pick and the put-down are solved for that height on a thread (`pick_worker.py`), and the
motion solved before plays until a solve is done. UI 3's *Pick motion* card (the deploy repo) shows what plays, its
numbers, the solve's state and its clip, and sets the flags above.

## What differs from the page

The page simulates the robot. Here the robot is measured, a little late and with some error, and these differ
(each a default of `mm_driver.py`):

- `--lookahead 60` (the page: 25). The node plays a streamed frame only once 46 frames follow it.
- The seed compares the SLAM pelvis with the reference frame of the pose's own time plus `--track-delay 0.1` s (the
  page: the frame tracked now). It turns all of the yaw gap, as the page does, since 2026-11-06 (`--yaw-gain 1`, 0.3
  before). Before POSE it moves the controller onto the robot with all of the yaw, as the loco driver does.
- In POSE each frame's heading is sent without the seeds' turns since POSE started. The deploy node takes the stream's
  heading at the start of POSE as the robot's and turns the robot with every change of it, so a seed sent with its
  turn turned the robot as much as the reference, and a yaw gap at the start of POSE never closed: both turned on
  forever.
- In planner mode the keys go to the node's planner (the page has none), turned from the map frame into the planner's
  own, whose +x is the robot's heading when the planner started (the node's heading state on `g1_debug`).
- The box is `live.sam3box`'s report, a few times a second and 0.2 to 0.6 s late. A move is taken after
  `--sync-need 2` reports that agree, the grip check after `--grip-need 2` reports.
- Since 2026-11-09 the reports move the box only while that can be trusted (`--box-lock on`, UI 3's *Box* tab): the
  median of `--box-lock-need 3` reports seen while the measured pelvis stands still in Locomotion (within 5 cm and
  3.4° for `--box-still-s 0.5`) locks it, and then no report moves it until the task leaves Locomotion
  (`--box-lock-moving on`); none ever moves it during Pick up, Lift, the carry or the put-down. Back in Locomotion the
  next stand measures it again. A pick asked for before the lock waits for it, up to `--box-lock-wait 3` s.
  `--box-freeze on` stops every update. The page has no such rule: its box sync (runner.js `syncBoxes`) reads the
  physics' box.
- The palm force aims at a fresh report of the box on the floor until the reference lifts the box, then at the box
  where the reference holds it (the page: the physical box always).
- In locomotion no seed puts the controller's simulation object inside a box's collision zone (the box's half size
  plus `collisionRadius` 0.36 m). The controller pushes the simulation object out of the zone at its next step, so
  such a seed is undone within 0.1 s. After a place the robot often stands 0.2 to 0.3 m closer to the box than the
  reference, and seeding it every 0.2 s made the reference shuffle in place until the robot kicked the box. While
  the robot stands that close, the reference stays where it is, up to 0.44 m behind the robot (`--keep-out off`
  turns the rule off).
- A grip-check drop clears the place target. On the page the drop comes during the rise, before any place is asked
  for. Here the D435i sees the box left on the floor some 2 s after the grasp, and a place asked for meanwhile made
  the controller put down the box right after its next lift, held or not.
- The page presses the palms only while the physical box is within 0.45 m of their midpoint (`graspStep`) and checks
  the grip on the physics' box at every step (`checkGrip`). The driver has no such box, so the measured palms take
  its place: palm sites closer together than `--empty-gap` (0.3 m, they sit 5.3 cm
  inside each hand, so the box held is about 0.44 m) with the palm force on hold no box. The force and the squeezes stop at once, before the hands
  meet, and 0.2 s later the hold is dropped as by the grip check. UI 1 and UI 3 draw the last 12 s of the gap
  (`grasp_trace`) and say what dropped each hold (`last_drop`: grip, palms, pose). Beside it, for display only, the
  driver asks the map server for its newest LiDAR scan in a vertical column at the palms' midpoint (the `column`
  op, 10 Hz): the box's top while the hands hold it. A hold disagrees when the column tops more than 0.05 m below
  the palms, or is empty where the LiDAR sees it at the palms' height (`grasp_trace.column`, `between` events, and
  each drop's `between_s`: how long before it the column disagreed). Nothing is dropped for it. The force stays off until the frames of the re-plan after the drop play: in the middle of a
  lift the controller grasps again on its first frame (the clip's held output is high and `graspReach` is horizontal
  only), as the page's controller does.
- Leaving POSE (`P`, `O`) drops the hold and clears the targets. Out of POSE the node's planner has the arms and the
  palm force is off, and the reference went on carrying with its hands together, which the next POSE went back to.
- UI 3's *Box task state* card draws the task as a state machine: the state at the played frame, the signals its
  transitions read, and the last 16 transitions with their causes (the driver's state, `fsm`).

## Parity with the page

```bash
node tools/page_parity/js_trace.mjs ~/Projects/lmm-g1-box /tmp/js.jsonl --states /tmp/states.jsonl
python tools/page_parity/check_parity.py /tmp/js.jsonl --forced /tmp/states.jsonl
node tools/page_parity/js_match_cases.mjs ~/Projects/lmm-g1-box /tmp/cases.json
python tools/page_parity/check_parity.py /tmp/cases.json --cases
node tools/page_parity/js_pickmotion.mjs ~/Projects/lmm-g1-box /tmp/pickmotion
python tools/page_parity/check_pickmotion.py /tmp/pickmotion
```

The forced check loads the page's whole state before each frame of the page's scripted sequence, steps once and
compares the state after it. A free run follows the page until rounding decides a near tie in the search, and from
there it takes another path of the same controller. In one test the costs were 4.1112789 and 4.1112797 at frame 484.

All checks pass against lmm-g1-box `v4.2.3` (`9c45ac7`, 2026-11-20), as they did against `v4.2.2` (`3d599cf`), `v4.2.1`
(`121379f`), `v4.2` (`b2a0a84`), `v4.0.1` (`114dcef`) and `v4.0` (`51f3eae`) and the first two against `v3.0`. What each
release changed for this port:

- `v4.2.3` brought two settings and changed no default, so the robot moves as with `v4.2.2`. The Pick up and the Put
  down query can take a weight for x and one for y of the box position and of the box direction, in the robot's root
  frame (x ahead of the robot, y to its left): a weight list of 9 values in place of 7, `[.., x, y, ..]` where the one
  weight of the group was (`AXIS_GROUPS`, `has_axis_weights`, `axis_weights`, `group_weights` in
  `mm_g1/controller.py`). The two values of a group are normalized together, so equal x and y weights are the one
  weight. And the force of the grasp heuristic can act on the wrists in place of the palms (`--force-at wrist`: the
  origin of each wrist yaw link, 10 cm before the palm site. `--palm-target mid` is then the midpoint of the two
  wrists). It still comes on by the midpoint of the palms. The force is the driver's (`PalmGrasp`), on the measured
  robot, so no parity check covers it: its torque on the left arm agrees with MuJoCo's `mj_jacBody` to 0 in a unit
  check, and with the force on the wrists the wrist yaw joints get no torque.
- `v4.2.2` brought the turn speed limit: the simulation object of Locomotion turns at most 60 deg/s
  (`simMaxTurnRate`, the driver's `--turn-rate`). It turns to the direction asked for with a spring, and a spring
  turns faster the larger the angle, so a turn to a target behind the robot turned the pelvis at about 300 deg/s on
  the page. The spring now sees at most `2 * limit / y` of the angle to its goal (`y`: its damping), and the
  trajectory of the query is stepped with the same limit. A smaller turn is the spring of before, and the box
  controller, which has no simulation object, is not changed. `--turn-rate 0` is the rule before. The rotation
  halflife of that spring (`simRotationHalflife`, 0.27 s as before) is the flag `--rot-halflife` now. No data
  changed. `mm_g1/mathq.py`'s limited spring agrees with the page's `math.js` within 9.3e-15 on 400 random
  springs of 30 steps (165 of them start above the limit's angle).
- `v4.2.1` changed only the data of the empty-handed Locomotion: the EMM takes `strafe` and `turn_in_place` are left
  out (`data/README.md`). No rule and no default changed, so the port's code is the same.
- `v4.1` (`a59750d`, 2026-11-17) is the release with new rules: the two states before a grasp and before a release,
  the walk that plays on into its SceneBot clip (`walkOn`), the bend that ends only by its event or by a cancel
  (`bendOut`), the foot lock that waits for a foot still in the air (`landHeight`), the put down's own manual edit,
  and a rise that ends at the end of its clip.
- `v4.2` moved the page's scripts into folders (`web/js/mm/`, `ui/`, `pages/`) and removed its Learned Motion
  Matching networks, which this port never ran. Of the data only `sbplace/config.json` changed (its `nlatent`, the
  networks' latent size, is gone).
- `v4.0.1` changed only the controller's defaults (the enter and leave thresholds and the pick's engage radius).

The numbers against `v4.2.3`, the same as against `v4.2.2` (with the turn speed limit the sequence is 43 frames longer
than `v4.2.1`'s, 4,923 frames then):

- The forced check: 4,966 frames, no discrete difference, the largest gap 5.3e-14 (the feet). The sequence
  plays *Stepping to the box* for 292 frames and *Bending down* for 594, and one pick and one put-down are cancelled
  in it (`cancel-moved`: the pick rises as a put-down rises, the put-down as a pick). Each of its put-downs is taken
  by distance inside the bend, so *Stepping to the marker* does not play. It takes box C from the page's table, so it
  runs the pick motion too. The free run follows the page for all of its 4,966 frames.
- Two feet wait in that sequence at the page's 0.02 m, 0.13 s in all (none with `v4.2.1`). With `--foot-land 0.002`
  on both sides 13 feet waited with `v4.2.1`, 0.90 s in all, and the forced check passed the same way (not run
  again since).
- A weight for x and one for y: the same sequence with `--w-pick 0.75,1,1,3,1,1.5,2.5,3,3 --w-place` the same on
  both sides (`js_trace.mjs` and `check_parity.py` take the two options) is another sequence, 4,982 frames. The
  forced check passes on it with no discrete difference and the largest gap 1.1e-13 (the feet), and the free run
  follows the page for all of its frames.
- `--cases` checks `Controller.match` alone on 5,408 searches made around the SceneBot pick and place, each one as
  Locomotion and as the box controller before its event (2,704 with `v4.0.1`, which had one rule for both). The
  playing frame is every 5th frame of each SceneBot range plus 60 random frames, and each query is a database
  frame's features plus noise. It also compares the tables the new rules read: the 200 frames where a SceneBot clip
  starts in a walk, the 60 walk ranges, the 562 frames a walk hands over at, and the 2,098 frames a bend rises at.
- The pick motion check runs the page's `mm/pickmotion.js` and `mm_g1/pickmotion.py` on the same boxes and table
  heights (12 cases: 4 of them are the put down's own edit, a bend that lends its edit to the other clip, and the
  manual edit read from its texts) and compares the solved rows of the Animation Database: the box is the same, the
  pelvis position within 9.1e-13, its rotation within 1.8e-12, the joints within 3.6e-15 and the velocities within
  7.5e-9 (MuJoCo 3.14.0 in the page's WASM, 3.3.3 here).

## Results

In the sim with the LiDAR, FAST-LIO2 and SAM3 (motionmatching-g1-deploy's `README_BOX_SIM.md` *Results* has the
tables): picking the box up and putting it down in front worked on the first try in all four runs, the box landing
0.06 to 0.20 m beyond the place target. When the box jumps 0.5 m back as the hands reach it, the regrasp is not
reliable: one run lifted it on the first regrasp, one on the second, four on the third to fifth crouch, and one not in
300 s. The regrasp bends down 0.5 to 0.9 m from the box, and the hands reach it from about 0.5 m. One run of each
scenario (pick9, shift7) ran the port of `1721681`, the others the port of `a41bd3f`. Untested on the robot.

With lmm-g1-box v4.2.1 (2026-11-18, `README_BOX_SIM.md` *Results*): with empty hands the robot's torso stood 10 deg back
after POSE started with the data before (on `turn_in_place`) and stands 4 deg forward with v4.2.1's (on
`basic_locomotion_1`). It strafed with its torso 6 to 10 deg back (on `strafe`) and strafes with it 5 to 7 deg forward
(on `locomotion_loops_2`), about as far in 3 s. The floor pick with a 3 s carry and the put-down worked in three runs of
three, each pick on the first crouch and each box upright after the put-down.

With lmm-g1-box v4.2 (2026-11-18, `README_BOX_SIM.md` *Results*): the floor pick with a 3 s carry and the put-down
worked in three runs of three with the defaults and in three of three with the rules before (`--walk-on off --bend-out
leave --foot-land off`), every pick on the first crouch. One box of the six ended on its side (a run with the
defaults), which three runs a side do not pin on the rules: the box hangs tilted in the hands in every run. `C` from
UI 3 while bending down made the robot rise empty-handed in 1.0 s, and `C` while lowering the box made it rise still
holding in 1.0 s, both with no search. The put-down on a 0.5 m table went as before (the box on the table, on its
side). A box that jumps 0.2 m back at the grasp was lifted on the sixth crouch in one run and on the third in another,
as unreliable as before. In both, one bend passed its grasp frame with no grasp, and the controller cancelled it and
took the pick again (`cancel-missed`, which the driver prints).

With lmm-g1-box v4.0.1's defaults (2026-11-16, `README_BOX_SIM.md` *Results*): the box controller takes a pick
0.55 to 0.59 m from the box (1.0 m with v4.0's), and the floor pick, the pick from a 0.5 m stand with the put-down 1 m
to the side (two runs, and two with v4.0's values) and the put-down on a 0.5 m table went as with v4.0. A put-down 0.8 m
ahead of a box on a stand lies behind the stand, where the robot cannot stand: there the box dropped and the robot fell.

With lmm-g1-box v4.0 (2026-11-16, `README_BOX_SIM.md` *Results*): the floor pick and put-down worked with SAM3. With
the sim's true box, a box on a 0.5 m stand was picked with the pick motion solved for that height, and a box was put
down on a 0.5 m table whose height came from the map, where it landed on its side (the box hangs tilted in the hands,
as before the port). SAM3 on the D435i does not see a box on a 0.5 m stand from 2 m: the image's top is 26° below
level.

## Credits

- G1 model: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `unitree_g1` (license in
  `assets/unitree_g1/LICENSE`).
- Motion data and method: `data/README.md`.
