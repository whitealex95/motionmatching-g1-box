# SuperMimic and SONIC physical tracking results

480 box episodes completed: 20 OmniRetarget clips × 8 heuristics × 3 checkpoints.
Physical box offset is 0 m. Box mass is 0.25 kg, friction is 1.5, and each tracker uses its native PD gains.
SuperMimic uses the browser demo weights downloaded on 2026-10-21. The demo does not identify the paper training checkpoint.

| Grasp heuristic | SONIC release | SONIC v1.1 | SuperMimic demo |
|---|---:|---:|---:|
| no squeeze | 0/20 | 0/20 | 0/20 |
| shoulder 0.6 rad | 0/20 | 2/20 | 12/20 |
| wrist 0.2 rad | 1/20 | 0/20 | 0/20 |
| shoulder 0.6 + wrist 0.2 | 7/20 | 1/20 | 11/20 |
| palm 15 N | 11/20 | 10/20 | 17/20 |
| palm 25 N | 13/20 | 15/20 | 18/20 |
| palm 40 N | 14/20 | 16/20 | 17/20 |
| palm 25 N + shoulder 0.6 + wrist 0.2 | 14/20 | 6/20 | 16/20 |

Success requires a lift above resting height + 0.25 m, no fall, and box height above resting height + 0.15 m at the reference carry end.
The nine rendered clip runs exactly match their sweep records for success, fall, peak height, carry-end height, and final height.

## Locomotion

Same EMM autopilot, seed 0, snap-xy correction, 0.4 s reference lead, and 60 s duration cap. Physical feedback generates a separate reference stream for each run.

| Tracker | Duration | Fell | Mean joint error | Joint error p95 | Waypoint index |
|---|---:|---|---:|---:|---:|
| sonic | 60.0 s | False | 0.1090 rad | 0.1349 rad | 9/13 |
| supermimic | 60.0 s | False | 0.0645 rad | 0.1008 rad | 9/13 |

Both runs remained upright and reached waypoint index 9. Neither completed the 13-waypoint tour within 60 s. Both measured zero root penetration into the matcher obstacle regions.
This is one scene and one seed. Joint errors are measured against each run's own generated reference.

## Full box motion matching

The pick-here test uses the SceneBot box, 25 N palm force, zero shoulder and wrist squeeze, and anchor reference correction. SuperMimic completes pickup, carry, upright placement, and walking away in 26.9 simulated seconds. SONIC release remains upright but does not complete within 60 seconds.
The comparison video freezes SuperMimic's final frame after its episode ends. This separate test uses the motion-matching pipeline and a different box from the recorded-clip evaluation.

## Verification and provenance

- Native demo playback versus streamed references: 20 actor steps with zero maximum absolute joint-target difference.
- An altered engine checksum is rejected before inference.
- Desktop/mobile browser checks passed for video playback, task and checkpoint switching, and layout. The local server supports HTTP range requests for seeking.
- Complete per-episode values, commands, source/asset hashes, and environment details are in `box/comparison.json`. Locomotion commands and metrics are in `loco/comparison.json`.
- The sweep resumed after its execution session terminated. Completed groups were preserved and source/asset hashes were checked before resuming. Aggregate elapsed wall time includes interruption gaps. Each group has its own measured subprocess wall time.
- No training was performed.
