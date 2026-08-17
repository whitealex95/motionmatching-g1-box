# SceneBot demo assets (vendored)

Downloaded from the SceneBot interactive web demo
(https://ericcsr.github.io/scenebot/demo.html) on 2026-08-16. The demo is a
MuJoCo-WASM page; these are the exact files it streams from
`https://ericcsr.github.io/scenebot/mujoco_wasm/dist-desktop/scenebot/`.

| file | contents |
|---|---|
| `policy.onnx` | whole-body tracking policy, obs 160 -> actions 29 (Isaac order). sha256 `c0a3caf00809d1ccb26aad6358f2b107098462deccac755e406e522134103d04` (matches `policy_meta.json`) |
| `policy_meta.json` | obs layout, joint-order maps, PD gains, action scales, default pose, init qpos |
| `clips.bin` / `clips_index.json` | 12 reference clips at 50 Hz: joint pos/vel (29, Isaac order) + world body pos/quat (30 bodies, Isaac BFS order, wxyz), float32 |
| `contact_labels.bin` / `contact_labels_index.json` | per-frame contact labels (4-dim: L/R foot, L/R wrist; clip 9 has a 5th sit channel) |
| `motion_graph.json` | command graph over clip segments (Forward/Backward/turns/PickUpBox/...) |
| `scene_29dof_flat_hand.xml` | their demo scene: G1 + free box on a 0.36 m pedestal + steps |
| `assets/g1/` | Unitree G1 29-DoF "flat hand" model (rubber-hand visuals + flat palm collision boxes + palm sites) and meshes |

Local additions (not from the demo):

- `scene_box_floor.xml` -- same G1, box on the floor, no steps (used by
  `scenebot_g1_box/run_pickup.py`)
- `scene_robot_only.xml` -- G1 + floor only (used by
  `scenebot_g1_box/run_mm_pickup.py`, which adds the carton in code)

The box pickup motion is clip 11 (`stitched_motion.npz`), frames 0..120:
a squat pickup of a box whose centre sits ~0.29 m ahead and ~0.19 m up from
the robot's standing position, hands 0.41 m apart. `PickUpBox` plays it
forward at half speed; `PutDownBox` plays the same frames backward.

Credit: "SceneBot: Contact-Prompted General Humanoid Whole Body Tracking
with Scene-Interaction" -- Sirui Chen, Shibo Zhao, Zhen Wu, Jiaman Li,
et al. (https://ericcsr.github.io/scenebot/). These files are kept here
only to reproduce the demo offline in this research sandbox.
