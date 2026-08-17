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

## run_mm_pickup.py — this repo's motion-matched pickup

The `mm_g1` matcher's box pick/carry/place reference (OmniRetarget carton
clips, streamed by `sonic_g1_box/mm_stream.py`) tracked by the SceneBot
policy instead of SONIC. Each 50 Hz frame becomes a SceneBot packet: leg
joint targets, VR 3-point targets via FK on the SceneBot G1, anchor = the
reference root, and a synthesized contact label (feet during pick/place,
wrists while reaching or holding). The commander steers the matcher by the
reference root; the policy follows through the anchor-error observation.

This reference is OUT of the SceneBot policy's training distribution. Two
adaptations make it work, both borrowed from how SceneBot treats its own
skill: the deep-squat portion is tracked at half rate (their demo plays its
pickup clip at `pickupForwardStepScale = 0.5`), applied only while the
reference is near-stationary, and the commander pauses after the place
before walking away.

| mode | box | result |
|---|---|---|
| `kinematic` (default) | no collision, teleported to the reference | SUCCESS — full sequence, no fall; transient lag up to ~0.75 m where the lift translates fast |
| `weld` | free carton, welded to the pelvis at the reference box-in-pelvis pose while held | SUCCESS — 0.5 kg carton genuinely loads the robot (lifted to ~1.0 m), max error 0.47 m |
| `grasp` | free carton, hand friction + `--squeeze` | FAILS — the arms do not hug the 0.34 m carton tightly enough on this out-of-distribution reference; raising arm gains topples the robot |

```bash
~/miniconda3/envs/mm-g1-sonic/bin/python run_mm_pickup.py                # kinematic
~/miniconda3/envs/mm-g1-sonic/bin/python run_mm_pickup.py --mode weld
```

## Comparison with SONIC (sonic_g1_box)

SONIC never observes the reference's xy position, so that stack needs the
ref-mode machinery to keep the reference pinned to the robot. The SceneBot
policy observes the anchor position error directly, so the reference can
simply lead and the robot follows — no replanning or anchoring needed —
but it tracks its own clip style much more tightly than the OmniRetarget
style (loose arm tracking is why the frictional grasp fails here while
SONIC's `run_grasp` succeeds with squeeze).

The amber stick figure is the reference the policy is tracking; in weld
mode the amber box outline is the reference box.
