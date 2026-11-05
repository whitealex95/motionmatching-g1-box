# SuperMimic physical tracking

This adapter runs the actor from the [SuperMimic browser demo](https://supermimic.github.io/demo/)
against native MuJoCo state. MuJoCo simulates the robot and box. The learned
world model is loaded as part of the demo bundle but does not advance physics.

The downloaded artifact is the demo's `weights.bin` plus `engine.wasm` and
`engine.cfg`. It is not an ONNX or PyTorch checkpoint. The demo does not identify
which paper training checkpoint it represents. Results must be described as
**SuperMimic demo weights, downloaded 2026-10-21**. File sizes, source URLs, and
SHA-256 checksums are in `assets_manifest.json`.

## Setup

Requires Node.js 18+, NumPy, MuJoCo, and the existing SONIC Python environment.
From the box repository:

```bash
python3 tools/supermimic_tracking/fetch.py
node tools/supermimic_tracking/runtime.mjs assets/supermimic/demo --self-test
```

The fetcher verifies existing files and refuses a changed upstream bundle.
The runtime also verifies its inputs before using its version-specific interface.
Set `SUPERMIMIC_ASSETS` to use another local directory containing the same files.

## Integration

`SuperMimicPolicy` accepts the projects' existing reference-motion interface:
joint positions and velocities in IsaacLab order, root position, and wxyz root
quaternion. It converts joint order, applies the existing heading alignment,
and computes reference body transforms with the demo G1 model. Each step sends
the physical robot's 36 qpos and 35 qvel values to the demo actor. The actor
returns 29 absolute joint targets in MuJoCo order.

The adapter supplies a 46-frame reference window. The actor samples ten frames
at stride five, covering 0.9 seconds at 50 Hz. Windows clamp to the newest
available frame. Paused playback holds the pose and zeroes reference joint
velocities. Reset starts from rest. The actor retains its history across control
ticks and replans. Reference body velocities used during reset are zeroed so
the bundled initialization motion does not affect custom motion reset.

The demo has no custom-motion upload API. `runtime.mjs` uses a checksum-pinned
memory layout to install caller-authored reference arrays and set the reference
playhead. Inference remains in the original WebAssembly engine. This is an
experimental compatibility adapter, sensitive to upstream engine changes.
The self-test compares 20 successive native demo actor steps against rolling
reference windows and requires absolute target error at most 1e-6.

Each tracker uses its native PD gains. SuperMimic gains and torque limits come
from the downloaded `scene.xml`. The physical robot geometry and inertial
parameters remain those selected by the project runner. The box evaluation uses
the SceneBot robot, frictional carton, and existing success criterion. These
experiments measure transfer of the demo actor to the project scenes.

## Run the physical variants

```bash
# Box motion matching and frictional grasp
MUJOCO_GL=egl ~/miniconda3/envs/mm-g1-sonic/bin/python \
  sonic_g1_box/run_grasp.py --tracker supermimic --no-video \
  --pick-here --palm-force 25 --shoulder-squeeze 0 --wrist-squeeze 0

# All 20 recorded box clips
MUJOCO_GL=egl ~/miniconda3/envs/mm-g1-sonic/bin/python \
  sonic_g1_box/run_track_clips.py --tracker supermimic --box-offset 0 \
  --palm-force 25 --shoulder-squeeze 0 --wrist-squeeze 0

# Locomotion, from the sibling motionmatching-g1-loco repository
MUJOCO_GL=egl ~/miniconda3/envs/mm-g1/bin/python -m sonic_tracking.interactive \
  --scene 00_maindemo --controller emm --tracker supermimic --auto \
  --ref-mode snap-xy --record /tmp/loco_supermimic.mp4 --max-seconds 60
```

Locomotion imports this shared adapter from the sibling box repository. Set
`SUPERMIMIC_TOOLS=/absolute/path/to/motionmatching-g1-box/tools` if they are
not siblings. Omitting `--tracker` retains SONIC. Hardware deployment is unchanged.

Full evaluation, rendering, and the local comparison page are under
`eval_supermimic/`. `SONIC_ORT_THREADS` optionally controls ONNX Runtime CPU
threads per session. Its absence preserves the previous session defaults.
