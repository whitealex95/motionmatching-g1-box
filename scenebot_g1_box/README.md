# scenebot_g1_box — the box pickup tracked by SceneBot's policy

The whole-body tracking policy and G1 model from the
[SceneBot](https://ericcsr.github.io/scenebot/) interactive web demo,
downloaded from the demo page and ported to Python
(`tools/scenebot_tracking/`, assets in `assets/scenebot/`). Two runners use
it to track a box pickup in full MuJoCo physics.

```
policy      single ONNX MLP, obs 160 -> 29 actions (Isaac order), 50 Hz
control     torque PD at 200 Hz (gains/limits from policy_meta.json)
obs         leg command (24) + VR 3-point wrist/head targets (21) + anchor
            pose error (9) + proprioception (96) + 10-way contact mask (10)
```

## run_pickup.py — SceneBot's own pickup motion

Faithful port of the web demo, minus the browser: the vendored motion graph
streams the reference (walk clips, squat pickup = clip 11 frames 0..120 at
half speed, put-down = the same frames backward), and their `free_box`
(0.3 x 0.2 x 0.3 m, 0.1 kg) is picked off the floor by hand friction alone.
After the pickup completes, the VR wrist targets and wrist contact labels
freeze (as in the demo) so the robot walks while carrying.

A scripted commander replaces the demo keyboard: settle, walk ~2 m to the
box, pick (G), hold, carry-walk, put down (P), retreat. The reference
stream is open loop, so a physics-free dry run of the same script predicts
the pickup pose, and the box is spawned exactly where the clip's hands
close (0.29 m ahead, computed from clip 11).

```bash
~/miniconda3/envs/mm-g1-sonic/bin/python run_pickup.py           # -> out/*.mp4
~/miniconda3/envs/mm-g1-sonic/bin/python run_pickup.py --viewer  # watch live
```

Exit 0 only if the box was lifted (> 0.4 m), placed back flat at rest
height, and the robot walked away without falling. Measured: full sequence
in ~24 s sim / ~5 s wall, box carried at 0.87 m.

## run_sequence.py — the demo page's Enter sequence

Replays the web demo's automatic example sequence Q-L-L-E-W-N-G-W-Z-P on
their original pedestal scene: spin left, sit on the bench, stand up, spin
right, walk, step onto the 0.36 m pedestal, grab the box off it, carry it
across the top (upper body frozen), step down still carrying, put it on
the floor. Token semantics match the demo's example runner (wait for graph
idle + 0.4 s settle, then tap; Q/E queue +/-180 deg of reference yaw at
60 deg/s; L toggles sit/stand). The reference stream is open loop, so this
reproduces the web demo's reference trajectory exactly.

```bash
~/miniconda3/envs/mm-g1-sonic/bin/python run_sequence.py    # -> out/scenebot_sequence.mp4
```

Measured: the full sequence succeeds first try in ~40 s sim (sat at root
z 0.57, pedestal at 1.16, box lifted to 1.27 m, placed flat on the floor,
no falls).

All three runners overlay the live contact prompt (`contact_viz.py`): one
sphere per prompted link (feet, palms, pelvis), GREEN = terrain contact,
MAGENTA = object contact. The 10 mask slots are (link, scene-type) pairs
per the paper — K = {L/R foot, L/R wrist, pelvis} x {terrain, object},
even slot terrain, odd slot object — so the overlay shows exactly what
the policy's contact observation says each tick: green pelvis while
seated, single green feet during the pedestal climb, magenta wrists while
reaching and carrying, and nothing on the feet during plain walking
(zeroed on purpose).

## run_mm_pickup.py — this repo's motion-matched pickup

The `mm_g1` matcher's box reference (streamed by `sonic_g1_box/mm_stream.py`)
tracked by the SceneBot policy. On this branch the pick and the drop are the
policy's OWN squat pickup — clip 11 baked at the demo's playback rates
(half-speed forward, full-speed reversed; see the top-level README) — so the
old half-rate tracking adaptation is off, and the physical box is SceneBot's
0.3 x 0.2 x 0.3 m free box (0.1 kg). Only the carry frames (OmniRetarget)
remain out of the training distribution.

Each 50 Hz frame becomes a SceneBot packet: leg joint targets, VR 3-point
targets via FK on the SceneBot G1, anchor = the reference root, and the
contact label INHERITED from the library (`lib['contact']`): the demo's own
per-frame labels on the pick/drop frames — feet through the squat, wrists
anticipatory from the start of the reach (clip frame 40, ~1.5 s before
touch) — and wrists-while-attached on the carry frames. The synthesized
FK-based labels remain as a fallback for libraries without baked labels.

The commander no longer scripts the walk to the box: it presses B once
settled and the matcher's own MOVE-TO-PICK state plans the route, walks it,
settles at the stance, and enters the pick (approach heuristics from
motionmatching-g1-shelf).

| mode | box | result (this branch) |
|---|---|---|
| `kinematic` (default) | no collision, teleported to the reference | SUCCESS — full sequence in ~25 s sim, max xy err 0.38 m, no fall |
| `grasp` | free box, hand friction only | SUCCESS with NO extra knobs (`--arm-gain 1 --squeeze 0`) — the clip's hand spacing fits its own box, friction-only pick to 0.93 m, place flat, walk away |

```bash
~/miniconda3/envs/mm-g1-sonic/bin/python run_mm_pickup.py                # kinematic
~/miniconda3/envs/mm-g1-sonic/bin/python run_mm_pickup.py --mode grasp --arm-gain 1 --squeeze 0
```

(The old carton-based grasp needed `--box-scale 0.85 --squeeze 0.5
--arm-gain 4` because the OmniRetarget palms hover 4-8 cm off the box faces;
see the git history of this README for that sweep.)

### run_mm_interactive.py — drive the physical robot with the keyboard

The same pipeline, interactive: a GLFW window shows the physics scene and
the keyboard steers the REFERENCE matcher live (open loop; the policy
follows through the anchor-error observation). The reference stream runs
only 2 frames ahead of the playhead (the scripted runner buffers 0.4 s),
so key response is immediate.

```bash
~/miniconda3/envs/mm-g1-sonic/bin/python run_mm_interactive.py             # grasp (default)
~/miniconda3/envs/mm-g1-sonic/bin/python run_mm_interactive.py --mode kinematic
~/miniconda3/envs/mm-g1-sonic/bin/python run_mm_interactive.py --script    # headless regression
```

WASD moves (camera-relative, Shift = walk pace), arrows face, B walks over
and picks up / sets down, T toggles the gizmos (green route, red command
taps, contact prompts), Esc quits. Needs a display; `--script` replays a
fixed timeline headlessly instead (settle, B, carry-walk 3 s, B, retreat)
and is the regression test — it also exercises carry-walking, which the
scripted runner never does. No reset key: restart the script.

Building this exposed an approach bug: the reference used to stop hard at
the stance, and the physical robot (which lags decelerations) overshot and
kicked the box away before the squat — worst at the short stream margin
(0.53 m kick). The move-to-pick endgame now brakes earlier and arrives
slower (cap 0.55 m/s, release at 0.18 m), and the full grasp sequence
succeeds at every margin.

### stress_single.py — dynamic-box envelope of the single pick

Sweeps the PHYSICAL box's size and mass through the full grasp sequence
(36 runs; the reference motion and reference box stay data-sized). Metrics
in `out/stress_single/results.json`, one 640x360 video per run with the
verdict in the filename, figures + table from `stress_single_plots.py`.
Success = picked, carried, set down FLAT at rest height, no fall, walked
away. Measured envelope (base 0.30 x 0.20 x 0.30 m, 0.1 kg; 25/36 runs,
after the softened approach endgame):

- **scale**: 0.8x-1.6x ALL succeed; 0.5-0.6x are below the hands' closing
  width and never grip
- **mass**: 0.1-2 kg carried reliably -- the box rides lower in the hands
  as it gets heavier (max box z ~1.0 -> 0.65 m); 3 kg is the edge (flips
  between success and a fall across runs of nearby configs); 5 kg fails
- **grip width** (y, the hands close on +-y): 0.15-0.40 m fine; 0.10 m
  lifts but slips out mid-carry
- **height**: 0.1-0.3 m fine; 0.5 m lifts but tips at set-down (rests
  higher than the reference box); 0.7 m topples the robot
- **depth** (x, toward the robot): 0.30-0.50 m works; 0.10-0.20 m puts the
  faces where the hands never land
- **size x mass**: 1.0x and 1.2x boxes carry up to 2 kg; the 0.8x box gets
  marginal above 0.5 kg (carried low, occasionally not walked away from)

Still contact-rich at the edges: configs near a boundary flip outcome
between otherwise-identical runs.

### Grip closed loop (grasp with a ref mode)

Ported from `sonic_g1_box/demo_base.py`: while the box is loose, the
physical box pose is copied into the matcher's belief every tick, so a
retry reaches for where the box actually is. Once the matcher believes the
box is held, the grip is judged at the tracked frame — reference box
lifted (> 0.10 m) while the physical box stayed down (< rest + 0.04 m)
drops the matcher back to locomotion and the pick retries after
`--retry-cooldown` (default 2 s; an immediate re-squat topples the robot).
A placement only latches if the box was physically carried, so an abort
out of PLACE falls through to a retry instead of a walk-away.

The loop needs a ref mode because it assumes the reference world matches
the physical one; with `--ref-mode none` they drift, so open loop stays
single-shot (a slipped grasp pantomimes to the end and reports
INCOMPLETE).

### --ref-mode: closing the loop between reference and robot

By default (`--ref-mode none`) the matcher runs open loop, steered by its
own root, and the policy follows through the anchor-error observation. The
sonic_g1_box ref modes (`ref_modes.py`) are also wired in; with any of them
the commander steers by the PHYSICAL robot instead. Measured (0.5 kg box):

| ref-mode | result |
|---|---|
| `none` (default) | kinematic box SUCCESS with transient lag up to ~0.6 m |
| `snap-xy` | SUCCESS; best overall — tightest tracking with the squat preserved |
| `snap-xyyaw` | no fall, tight tracking (0.13 m), but the squat comes out shallow (borderline) |
| `snap-all` | tracks in 0.12 m but re-seeds the joint offsets from the lagging robot, which dilutes the squat away — the pickup degrades to a pantomime |
| `anchor`, `anchor-replan` | robot falls (tested gains 0.1 / 0.2, 0.738) — the continuous drift correction fights this policy |

`snap-all` is SONIC's default and works there because SONIC tracks the
squat tightly enough that re-seeding from the robot ~= the clip pose; the
SceneBot policy lags the out-of-distribution squat, so re-seeding keeps
resetting its progress. Use `snap-xy` here. `--anchor-gain` and
`--replan-gain` tune the two families as in sonic_g1_box.

## Comparison with SONIC (sonic_g1_box)

SONIC never observes the reference's xy position, so that stack needs the
ref-mode machinery to keep the reference pinned to the robot. The SceneBot
policy observes the anchor position error directly, so the reference can
simply lead and the robot follows — `--ref-mode none` works, and
`--ref-mode snap-xy` merely tightens it. It tracks its own clip style much
more tightly than the OmniRetarget style (loose arm tracking is why the
frictional grasp fails here while SONIC's `run_grasp` succeeds with
squeeze — no ref mode changes that, since the failure is in the arms, not
the root).

The amber stick figure is the reference the policy is tracking; in grasp
mode the amber box outline is the reference box.
