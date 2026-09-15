# EMM clips (Unitree G1, retargeted)

Clips from [Environment-aware Motion Matching](https://github.com/), retargeted to the
G1 and exported as MuJoCo qpos. Copied here so this repo stays self-contained.

    clips/<stem>.npz          the motion
    clip_videos/<stem>.mp4    kinematic playback preview

Source of truth: `~/Projects/Environment-aware-Motion-Matching/resources/g1/{clips,clip_videos}/`
(originally exported from `~/Projects/GenoViewPython-MotionMatching/resources/g1/emm_clips/`).

## Format

`npz` keys: `qpos` (T, 36) float32, `fps` (int, 30), `source` (the original BVH path).

    qpos = [ root_pos (3) m | root_quat wxyz (4) | dof (29) ]

This is exactly the 36-D layout `assets/unitree_g1/g1.xml` uses: freejoint + 29 hinges in
the standard Unitree G1 order (12 leg, 3 waist, 14 arm/wrist). Verified joint-for-joint
against EMM's `g1_mocap_29dof.xml`, so no reordering is needed and EMM's model and meshes
(52 MB) do not have to be vendored. Z-up, floor at z = 0, pelvis NOT ground-projected.

```python
import numpy as np, mujoco
q = np.load('clips/emm_extra__box.npz')['qpos']          # (T, 36)
m = mujoco.MjModel.from_xml_path('../../assets/unitree_g1/g1.xml')
d = mujoco.MjData(m)
d.qpos[:] = q[t]; mujoco.mj_forward(m, d)                # kinematic playback
```

## Clips

### `emm_extra__box`

5273 frames, 175.8 s at 30 fps, one continuous take, no trims and no mirroring.
Source BVH: `Unity/Assets/EnvironmentMotionMatching/Animations/BVH/emm_extra/box.txt`.

**This clip contains no box and no box carrying**, despite the name. Measured over the
full take:

- `qpos` is robot-only (36-D). There is no object channel, unlike the OmniRetarget
  `data/robot_object_g1/` clips, which are 43-D (robot 36 + box freejoint 7).
- Root z stays within 0.754..0.845 m for all 175.8 s. A pick or a place needs a squat,
  and none occurs.
- The lowest foot never leaves 0.036..0.099 m, so the character is never airborne and
  never stands on anything.
- Hands are ~0.58 m apart and hang loose in the frames that most resemble a two-handed
  hold. A geometric "both hands forward, level and symmetric" test flags 17% of frames,
  but those are mostly overhead arm raises, which pass the same symmetry test.

It is usable as **locomotion** data (the root travels ~6 m x 4 m at 0.43 m/s mean,
1.07 m/s max), not as `carry` data for the box state machine.
