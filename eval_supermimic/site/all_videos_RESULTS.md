# All OmniRetarget box recordings

720 measured and recorded episodes: 20 clips × 9 heuristics × 4 controller settings.
Each cell below reports successful carries out of 20.

| Controller | none | shoulder | wrist | shoulder+wrist | palm15 | palm25 | palm40 | palm60 | palm25+biases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SONIC release | 0 | 0 | 1 | 7 | 11 | 13 | 14 | 14 | 14 |
| SONIC v1.1 | 0 | 2 | 0 | 1 | 10 | 15 | 16 | 15 | 6 |
| SuperMimic, clip after force | 0 | 12 | 0 | 11 | 17 | 18 | 17 | 18 | 16 |
| SuperMimic, clip before force | 0 | 12 | 0 | 11 | 17 | 18 | 17 | 18 | 15 |

Palm force is specified per labelled palm. Each virtual force points toward the box centre, ramps over 0.15 seconds, and is converted to arm torque with the palm Jacobian.
The new SuperMimic row clips PD torque to the existing demo and scene limits before adding virtual-force torque. MuJoCo motor control, actuator force, and joint actuator-force clips are disabled for this row. The resulting applied torque can exceed these limits. The original SuperMimic and SONIC rows retain their existing torque paths.
Success requires peak box height above resting height + 0.25 m, no robot fall, and box height above resting height + 0.15 m at the reference carry end. The carton mass is 0.25 kg, friction is 1.5, and physical offset is 0 m.
Videos are 1280 × 720 H.264 at 25 fps. Policy control remains 50 Hz, with 20 physics substeps per control tick. Each recording includes one second of settling. Runs stop if the robot falls.

All 480 original baseline episodes exactly reproduce their stored outcomes, peak height, carry-end height, final height, and fall frame.
All 80 SuperMimic episodes without virtual force exactly match between clipping orders.
58 new-row episodes exert joint actuator torque above the demo limit. The largest measured excess is 4.144350 N·m. Diagnostics use MuJoCo qfrc_actuator at every physics substep, including settling.
All 720 video files passed container, frame count, resolution, codec, and frame-rate checks. video_manifest.json records file sizes, durations, frame counts, and SHA-256 hashes.
comparison.json records commands, dependencies, source and asset hashes, per-episode values, and wall time. No training was performed.
