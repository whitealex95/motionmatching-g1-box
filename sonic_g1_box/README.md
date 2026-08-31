# sonic_g1_box — motion matching + SONIC tracking

The box pick/carry/place motion from `mm_g1` as a kinematic reference,
tracked by NVIDIA's SONIC policy in full physics. Ported from
`motionmatching-g1-door/sonic-g1-door`.

```
reference   mm_g1 matcher, 30 Hz -> resampled 50 Hz stream (mm_stream, robot+box)
tracking    SONIC encoder/decoder ONNX (encode mode 0), 50 Hz
physics     PD + MuJoCo, 200 Hz; NVIDIA 29-DoF G1 scene + the large box
```

The commander presses B: the matcher's own MOVE_TO_PICK plans and walks the
approach from the live box pose, the pick/carry/place run, then the
commander walks the robot away. Decisions use the physical robot;
`--ref-mode` (default `snap-all`) keeps the reference root on the robot
(see `ref_modes.py`). Exit code 0 only if the box was lifted, placed back
upright at its resting height, and the robot walked away without falling.

Two variants, one file each (weld was removed — only kinematic tracking and
the frictional grasp remain):

| variant | file | box handling |
|---|---|---|
| (a) kinematic | `run_kinematic.py` | no collision, box teleported to the reference pose |
| (b) grasp | `run_grasp.py` | free body; friction only. While a hand's contact label is on, `--shoulder-squeeze` biases that shoulder roll inward and `--wrist-squeeze` toes the wrist yaw in. Contact-rich and jagged: small parameter changes flip the outcome |

```bash
~/miniconda3/envs/mm-g1-sonic/bin/python run_kinematic.py           # -> out/*.mp4
~/miniconda3/envs/mm-g1-sonic/bin/python run_grasp.py --no-video
```

Measured (defaults, headless, 0.5 kg box everywhere, SONIC release
checkpoint): kinematic succeeds in ~17 s. The frictional grasp of the
single-SceneBot squat pick currently SLIPS with the default tuning (box
reaches ~0.36 m, the grip check catches it, immediate retry, no fall over
60 s) -- tuning it, and trying the other SONIC checkpoints, is open work
on this branch.

`--sonic {release,low_latency,sonic_v1_1}` picks the SONIC checkpoint
(default `release` = v1.0). The variants live in
`assets/sonic/policy/<variant>/`; the two non-release decoders are ~150 MB
and gitignored, so run `assets/sonic/policy/fetch_models.sh` once to
download them.

The robot model defaults to SceneBot's flat-hand G1
(assets/scenebot/scene_robot_only.xml) — the model the motion and the
contact labels were made with, using its own flat palm collision boxes.
`--robot sonic` swaps in NVIDIA's 29-DoF SONIC scene instead; its rubber
hands are visual-only, so capsule contact pads are bolted onto the wrists
(a different contact geometry than the demo's). With either robot the
frictional grasp currently slips with default tuning.

A grip check closes the loop between the kinematic reference and physics:
the matcher's box is clip-driven once its data marks it held, so if the
reference box has lifted (at the tracked frame) while the physical box
stayed on the ground, the matcher's belief is dropped -- back to locomotion,
box fed back from physics, immediate retry -- instead of pantomiming the
whole carry and place with an empty-handed robot.

The physical box defaults to the SceneBot free box the motion library is
baked for, sized from the single source of truth `C.BOX_HALF`
(mm_g1/config/library.py) — the same setting run.py's interactive scene
uses. `--box carton` swaps in the printed MEDICINE carton mesh instead
(closed 24-vertex cuboid, 0.32 x 0.34 x 0.36 m — a DIFFERENT size than the
library's box; the carton sits tilted inside its local frame on purpose,
the OmniRetarget clips' box quaternion is calibrated to the scanned box's
frame, tools/make_medicine_box.py).

The amber stick figure is the reference frame the policy is tracking; in the
grasp variant the amber box outline is the reference box, so tracking
error and slip are visible.
