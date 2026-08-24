# sonic_g1_box — motion matching + SONIC tracking

The box pick/carry/place motion from `mm_g1` as a kinematic reference,
tracked by NVIDIA's SONIC policy in full physics. Ported from
`motionmatching-g1-door/sonic-g1-door`.

```
reference   mm_g1 matcher, 30 Hz -> resampled 50 Hz stream (mm_stream, robot+box)
tracking    SONIC encoder/decoder ONNX (encode mode 0), 50 Hz
physics     PD + MuJoCo, 200 Hz; NVIDIA 29-DoF G1 scene + the large box
```

A scripted commander walks to the box, picks it up (B), holds it, sets it
down, and walks away. It steers by the physical robot; `--ref-mode` (default
`snap-all`) keeps the reference root on the robot (see `ref_modes.py`).
Exit code 0 only if the box was lifted, placed back upright at its resting
height, and the robot walked away without falling.

Two variants, one file each (weld was removed — only kinematic tracking and
the frictional grasp remain):

| variant | file | box handling |
|---|---|---|
| (a) kinematic | `run_kinematic.py` | no collision, box teleported to the reference pose |
| (b) grasp | `run_grasp.py` | free body; friction only, `--squeeze` biases the shoulder rolls inward while held (contact-rich and jagged: small parameter changes flip the outcome) |

```bash
~/miniconda3/envs/mm-g1-sonic/bin/python run_kinematic.py           # -> out/*.mp4
~/miniconda3/envs/mm-g1-sonic/bin/python run_grasp.py --no-video
```

Measured (defaults, headless, 0.5 kg box everywhere): kinematic succeeds in
~13 s; grasp (arm-gain 6, squeeze 0.4) slips and retries but succeeds in
~24 s.

`--robot scenebot` swaps in the SceneBot flat-hand G1
(assets/scenebot/scene_robot_only.xml) instead of the NVIDIA model.
Kinematic (12 s) still succeeds; the
frictional grasp does NOT — the flat palm pads protrude ~1.4 cm less than
the bolted-on capsule pads and need face-parallel wrist alignment that
SONIC's arm tracking doesn't deliver, so the box slips on every attempt
(swept squeeze 0.4-0.7, arm-gain 6-8, `--box-scale` 0.85-1.0).

A grip check closes the loop between the kinematic reference and physics:
the matcher's box is clip-driven once its data marks it held, so if the
reference box has lifted (at the tracked frame) while the physical box
stayed on the ground, the matcher's belief is dropped -- back to locomotion,
box fed back from physics, immediate retry -- instead of pantomiming the
whole carry and place with an empty-handed robot.

The box is ONE geom: the printed MEDICINE carton mesh (closed 24-vertex
cuboid, 0.32 x 0.34 x 0.36 m) is both the collision shape and the visual.
The carton sits tilted inside its local frame on purpose — the clips' box
quaternion is calibrated to the scanned box's frame (tools/make_medicine_box.py).

The amber stick figure is the reference frame the policy is tracking; in the
grasp variant the amber box outline is the reference box, so tracking
error and slip are visible.
