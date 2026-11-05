# SuperMimic and SONIC comparison

The complete recording sweep tracks all 20 OmniRetarget clips with nine
heuristics, including 60 N per palm. Its four rows are SONIC release, SONIC v1.1,
SuperMimic with clipping after virtual-force addition, and SuperMimic with PD
clipping before virtual-force addition. All 720 episodes have individual videos
in `results/box_all_videos`. The earlier 480-episode sweep and nine sample videos
remain in `results/box` and `results/box_videos`.

Every condition uses the same scene, clip references, settle interval, box
properties, and success criterion. Physical box offset defaults to 0 m.
Pass `--box-offset 0.05` with a separate output directory to evaluate a perturbed
box. Videos use 1280 × 720 pixels at 25 fps. Policy control remains 50 Hz.

Success requires a peak box height above the initial resting height plus 0.25 m,
no fall, and a box height above resting height plus 0.15 m at the reference's
carry end. The last recorded box height can be low after a successful put-down.
These recorded-clip tests evaluate physical tracking and grasp retention.
The interactive box runner separately supports motion matching and planning.

Each tracker uses its native PD gains. The original SuperMimic row clips
combined PD and virtual-force torque to the demo limits, followed by the scene's
motor and joint actuator-force limits. SONIC retains its existing scene limits.
The added SuperMimic row uses `--torque-clip before-force`: PD torque is clipped
to the intersection of the demo and scene limits, then virtual-force torque is
added. The final scene motor control, actuator force, and joint actuator-force
clips are disabled for this row, allowing the sum to exceed the limits.

The 15, 25, 40, and 60 N settings are commanded virtual force per labelled palm.
The force points toward the box centre, ramps over 0.15 seconds, and maps to
arm torque through the palm Jacobian. It is not measured contact force.
`torque` episode records measure actual applied joint actuator torque from
MuJoCo `qfrc_actuator` at every substep, including settling. They include maximum
absolute torque, maximum excess over the demo limit, and the number of substeps
with excess. Limit-excess values apply to the two SuperMimic rows and use
the demo limits. In `sonic_g1_box/run_grasp.py`, combine the new option with
`--palm-force-via torque` to add virtual-force torque after the PD clip.

The downloaded demo's paper training checkpoint identity is unspecified.

```bash
# From motionmatching-g1-box
python3 tools/supermimic_tracking/fetch.py
node tools/supermimic_tracking/runtime.mjs assets/supermimic/demo --self-test

~/miniconda3/envs/mm-g1-sonic/bin/python eval_supermimic/test_torque_control.py
~/miniconda3/envs/mm-g1-sonic/bin/python eval_supermimic/run_comparison.py \
  --jobs 6 --video --video-fps 25 --out eval_supermimic/results/box_all_videos
python3 eval_supermimic/validate_all_videos.py
python3 eval_supermimic/build_site.py
python3 eval_supermimic/serve.py
```

Open <http://localhost:8766>. The page includes synchronized video controls,
a 20-clip selector, four controller rows, nine heuristic columns, all per-clip outcomes, and downloadable results. Click a table result to load its video. The scripts
refuse to overwrite completed evaluations. Choose a new `--out` for repeats.

The locomotion videos use the same EMM autopilot, seed 0, `snap-xy` reference
correction, 0.4 s reference lead, and 60 s duration cap. The reference stream
responds to the physical robot, so the two trajectories may diverge. Their
joint-error measurements follow each run's generated reference. These videos
are a single-scene comparison, not a locomotion benchmark.

JSON outputs include commands, individual outcomes, measured subprocess wall
time, source and asset hashes, and environment information. Wall time includes
startup, inference, physics, output, and rendering when enabled. Concurrent
jobs share CPU resources. No training is performed. The runners set
`SONIC_ORT_THREADS=2` and `OMP_NUM_THREADS=1` to bound CPU contention.

Videos and downloaded model assets remain Git-ignored. The page builder copies
videos to a local media directory, so the site can be served without exposing
the repositories or checkpoint files.

If a sweep session stops before completion, resume its unfinished groups with
`python eval_supermimic/run_comparison.py --jobs 6 --video --video-fps 25 --out eval_supermimic/results/box_all_videos --resume`. Keep the original arguments when resuming. The runner verifies the
recorded protocol and inference source/asset hashes before continuing.

The full box motion-matching recordings are `results/box_mm_sonic.mp4` and
`results/box_mm_supermimic.mp4`. Their exact commands and outcomes are in
`results/box_motionmatching.json`. The combined video freezes each panel after
its episode ends. The recorded-clip evaluation uses the frictional carton,
while this full pipeline test uses the SceneBot box.

The full recording report is `results/box_all_videos/RESULTS.md`. Its validation
checks all 720 video containers, hashes each video, reproduces all 480 prior
baseline episodes exactly, and compares the 80 episodes without virtual force
between clipping orders. `eval_supermimic/make_report.py` and `make_videos.py`
retain the earlier one-clip report and montage workflow.
