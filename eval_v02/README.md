# eval_v02 -- the squeeze test (v0.2)

Self-contained, reproducible evaluation of **grasp-strategy biases** for
tracking the 20 OmniRetarget G1 large-box carry clips with the SONIC tracker.

**What this measures.** Each clip is played back as a *kinematic reference* that
the SONIC tracker follows in MuJoCo physics with a frictional carton. The robot
and the box are initialised at the clip's first frame (the start of the pickup)
and the clip itself, resampled 30 -> 50 Hz, is the reference. **This is not the
full pick-carry-place pipeline**: there is no motion matching, no planner, and
no place phase beyond what the recorded clip already contains. The only thing
under test is whether the tracked robot keeps hold of the physical box while the
reference carries it.

## What each row means

Every run also carries the tracker defaults `--shoulder-open 0.4`,
`--box-friction 1.5`, `--box-offset 0.05`, `--box-scale 1.0`, `--substeps 20`,
`--settle 1.0`, `--speed 1.0`. The clip list is always all 20 stems from
`mm_g1.data._box_clip_names()` (listed at the end of this file).

| Condition key | Display label | Strategy CLI flags (added to `--box-mass 0.25`) |
|---|---|---|
| `none` | no squeeze | `--shoulder-squeeze 0 --wrist-squeeze 0 --palm-force 0` |
| `shoulder` | shoulder 0.6 rad | `--shoulder-squeeze 0.6 --wrist-squeeze 0 --palm-force 0` |
| `wrist` | wrist 0.2 rad | `--shoulder-squeeze 0 --wrist-squeeze 0.2 --palm-force 0` |
| `shoulder+wrist` | shoulder 0.6 + wrist 0.2 | `--shoulder-squeeze 0.6 --wrist-squeeze 0.2 --palm-force 0` |
| `palm15` | palm 15 N | `--shoulder-squeeze 0 --wrist-squeeze 0 --palm-force 15` |
| `palm25` | palm 25 N | `--shoulder-squeeze 0 --wrist-squeeze 0 --palm-force 25` |
| `palm40` | palm 40 N | `--shoulder-squeeze 0 --wrist-squeeze 0 --palm-force 40` |
| `palm25+biases` | palm 25 N + shoulder 0.6 + wrist 0.2 | `--shoulder-squeeze 0.6 --wrist-squeeze 0.2 --palm-force 25` |

Each condition runs for **both** SONIC checkpoints, all 20 clips:

| Checkpoint | Path | File | sha256 |
|---|---|---|---|
| `release` | `assets/sonic/policy/release/` | `model_encoder.onnx` | `013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3` |
| `release` | | `model_decoder.onnx` | `c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed` |
| `release` | | `observation_config.yaml` | `466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b` |
| `sonic_v1_1` | `assets/sonic/policy/sonic_v1_1/` | `model_encoder.onnx` | `fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae` |
| `sonic_v1_1` | | `model_decoder.onnx` | `34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509` |
| `sonic_v1_1` | | `observation_config.yaml` | `4a67713b310932e50aca81f19188c8d76013148e98b15c8b5bbea995f12e59f0` |

The `sonic_v1_1` decoder is gitignored (150 MB); fetch it with
`assets/sonic/policy/fetch_models.sh` before reproducing.

**Box parameters (identical in every condition).** Mass 0.25 kg, type `carton`,
sliding friction 1.5, scale 1.0. The *physical* box is shifted 0.05 m along the
robot's initial heading while the reference box stays at the recorded spot, so
every condition includes that fixed perturbation. The box is re-seated flush on
the floor at reset, and the robot's root z is lowered so the feet are flush.

### Why `palm25+biases` is a separate row

In `run_track_clips.py` the label-gated shoulder and wrist biases are applied
whenever they are non-zero, **independent of `--palm-force`**: the biases are
added to the tracker's joint target and the palm force is added to `data.ctrl`
as `tau = J^T f` on top. So running the script with `--palm-force 25` and
nothing else keeps the defaults `--shoulder-squeeze 0.6 --wrist-squeeze 0.2`
and measures *palm force plus both biases*. That is the code path behind the
August log `sonic_g1_box/out/track_clips/sweep_palmforce25.log` (13/20 release,
15/20 `sonic_v1_1`). To isolate each strategy, `palm15/palm25/palm40` zero both
biases explicitly and `palm25+biases` keeps the old combined path.

## Protocol

Copied from `sonic_g1_box/run_track_clips.py`:

- **Reference.** The clip's robot joints and box pose, resampled 30 -> 50 Hz with
  linear interpolation (quaternion slerp for the root and box orientation).
  Playback speed 1.0.
- **Control.** SONIC policy at 50 Hz (`CONTROL_DT = 0.02` s), 20 physics
  substeps per control tick, so the MuJoCo timestep is 0.001 s. Joint torques
  come from the stock PD gains `P.KPS`/`P.KDS`; `--arm-kp`/`--arm-kd` are 1.0
  (unscaled) everywhere here.
- **Squeeze biases.** While a hand-contact label of the clip is on, the target
  is offset by `+-shoulder_squeeze` on the shoulder-roll joints and
  `+-wrist_squeeze` on the wrist-yaw joints. With no contact label and the clip
  in the `PICK` or `PLACE` phase, the shoulders instead open by
  `shoulder_open = 0.4`.
- **Palm force.** For each labelled palm, `tau = J^T f` with `f` of magnitude
  `--palm-force` newtons pointing from the palm site toward the box centre,
  ramped in and out over 0.15 s.
- **Settle.** 1.0 s (50 ticks) with the policy paused at reference frame 0
  before playback starts.
- **Episode length.** `motion.timesteps + 1.0 s`, cut early on a fall.
- **Fall.** Robot pelvis `qpos[2] < FALL_Z = 0.28` m. The episode stops at that
  frame and the clip counts as `fell`.
- **Success (verbatim condition).** `lifted and not fallen and z_at_carry_end >
  z_rest + 0.15`, where
  - `z_rest` = median box z over the first `BOX_REST_FRAMES = 5` clip frames,
  - `lifted` = peak physical box z over the episode `> z_rest + 0.25`,
  - `carry_end` = the last 50 Hz frame at which the *reference* box z exceeds
    `z_rest + 0.25` (i.e. where the reference begins the put-down), and
  - `z_at_carry_end` = the physical box z at the first tick reaching
    `carry_end`.
- **Outcome labels.** `success` = the predicate above holds (the tracker's
  `SUCCESS`); `dropped` = lifted but not still held at `carry_end`;
  `slipped` = never lifted; `fell` = pelvis below 0.28 m.

"Carried" in the table and the plot is the count of `success` clips out of 20.

## Reproducing

```bash
cd /home/jkim3662/Projects/motionmatching-g1-box
# once, if assets/sonic/policy/sonic_v1_1/model_decoder.onnx is missing:
bash assets/sonic/policy/fetch_models.sh

MUJOCO_GL=egl ~/miniconda3/envs/mm-g1-sonic/bin/python \
    eval_v02/run_squeeze_sweep.py --jobs 8
~/miniconda3/envs/mm-g1-sonic/bin/python eval_v02/make_tables.py
~/miniconda3/envs/mm-g1-sonic/bin/python eval_v02/make_plot.py
```

The sweep runs one subprocess per (condition, checkpoint) -- 16 runs of 20 clips
each, 8 in parallel -- and writes

- `results/squeeze_results.json` -- the full record (per-clip outcomes, peak
  lift, reference peak, end z, fall frame, wall time, exact command),
- `results/squeeze_results.csv` -- one row per (condition, checkpoint),
- `results/logs/<condition>__<checkpoint>.log` -- raw tracker stdout with the
  command line at the top,
- `results/RESULTS.md`, `results/latex/squeeze_table.tex`,
- `results/squeeze_bar.png` / `.pdf` (and a dark-surface variant).

**Measured wall time** (this machine, 16 runs of 20 clips, 8 subprocesses in
parallel): the whole sweep took **936 s**. Per clip that is 12.4-29.4 s
(mean 20.1 s), `release` consistently faster than `sonic_v1_1` because the v1.1
decoder is 3.7x larger. Run alone, with no other process competing, one clip
takes about **4 s** end to end including process start and ONNX session load:
the per-clip numbers above are the cost of 8-way oversubscription on 32 threads,
not the cost of the simulation. Every run records its own `wall_seconds` and
`seconds_per_clip` in `squeeze_results.json`.

Everything is CPU-only: `run_track_clips.py` builds `SonicPolicy(device='cpu')`,
and the sweep additionally sets `CUDA_VISIBLE_DEVICES=''`. Rendering is off
unless `--video` is passed; `MUJOCO_GL=egl` is set for headless safety.

## Figure 2: three heuristics on one clip

`results/fig_box_heuristics.png` / `.pdf` -- a 3-row by 5-column filmstrip of
**one** clip under three grasp heuristics, all with the `release` checkpoint and
the same reference timestamps.

| Row | Label | Flags (plus `--box-mass 0.25`) | Outcome on this clip |
|---|---|---|---|
| 1 | no heuristic | `--shoulder-squeeze 0 --wrist-squeeze 0 --palm-force 0` | `slipped` (peak 0.20 m, reference peak 0.93 m) |
| 2 | joint squeeze | `--shoulder-squeeze 0.6 --wrist-squeeze 0.2 --palm-force 0` | `dropped` (peak 0.75 m) |
| 3 | virtual palm force | `--shoulder-squeeze 0 --wrist-squeeze 0 --palm-force 25` | `SUCCESS` (peak 0.88 m) |

**Clip: `sub12_largebox_071_original_mujoco`.** It is one of only two clips in
the sweep where the three setups give three *different* outcomes under the
`release` checkpoint (the other is `sub7_largebox_042_original_mujoco`), so the
figure separates all three without a tie.

**Columns (reference 50 Hz frame -> time after the settle phase).** `t = 0` is
the moment playback starts, i.e. the end of the 1.0 s settle, so video frame
index `= 50 + 50 t`.

| Column | Reference frame | t (s) | How it is chosen |
|---|---|---|---|
| settle end | 0 | 0.00 | the first playback tick |
| contact | 59 | 1.18 | first 50 Hz frame with a hand-contact label on |
| lift | 77 | 1.54 | first frame where the *reference* box z exceeds `z_rest + 0.25` |
| mid carry | 153 | 3.06 | midpoint between lift and put-down |
| put-down | 230 | 4.60 | last frame where the reference box z exceeds `z_rest + 0.25` |

The renderer draws the **reference** box as a translucent overlay (green while a
hand-contact label is on) next to the amber reference skeleton, while the
physical carton is the textured mesh. A slip is therefore visible directly: in
row 1 the green reference box rises to chest height while the brown carton stays
on the floor.

Reproduce with:

```bash
MUJOCO_GL=egl ~/miniconda3/envs/mm-g1-sonic/bin/python \
    eval_v02/run_heuristic_videos.py --clip sub12_largebox_071_original_mujoco
~/miniconda3/envs/mm-g1-sonic/bin/python eval_v02/make_plot.py --only strip
```

`run_heuristic_videos.py` runs the three tracker calls with `--video` one at a
time (they would otherwise overwrite each other's MP4, since the tracker names
the file from the clip stem and the variant only), moves each MP4 to
`results/videos/<clip>_<setup>.mp4`, extracts the five matched frames to
`results/frames/<setup>_<i>.png`, and records every command, path and timestamp
in `results/heuristic_frames.json`. `make_plot.strip()` then assembles the strip
from those PNGs alone, so the figure rebuilds without re-running physics.

**The three MP4s are not committed** (7.5-7.9 MB each, 23 MB together). The
repository's `.gitignore` already excludes rendered demo videos with the note
that they do not delta-compress and put hundreds of MB into `.git`, so
`results/videos/*.mp4` follows the same rule. They are on disk at
`/home/jkim3662/Projects/motionmatching-g1-box/eval_v02/results/videos/` and
are regenerated by the command above. The extracted frames (stills) *are*
committed, so the figure is reproducible from the repository alone.

## Relation to the August 2026 log

`sonic_g1_box/out/track_clips/sweep_palmforce25.log` (2026-08-31 14:41, four
minutes after commit `5930188`, which is still the current version of
`run_track_clips.py`) reports 13/20 for `release` and 15/20 for `sonic_v1_1`.
Its exact flags were not recorded.

The `palm25+biases` condition here reproduces **13/20 for `release`** -- the same
count -- but on a *different set of clips*, and it gives **6/20** rather than
15/20 for `sonic_v1_1`. Checked against that log:

- `run_track_clips.py`, `box_scene.py`, `tools/sonic_tracking/`, `mm_g1/labels.py`,
  the clip `.npz` files and their label sidecars, and the tracked `release`
  checkpoint are all **byte-identical** to the state at that commit.
- The conda environment has not been rebuilt since 2026-07-24.
- The pipeline is bit-reproducible on this machine: two serial repeats of
  `palm25+biases` on `release` give identical per-clip outcomes, and they also
  match the run made inside the 8-way parallel sweep.
- `--box-offset 0`, `--speed 0.5` and `--arm-kp 2.0` on top of `palm25+biases`
  with `sonic_v1_1` give 6/20, 2/20 and 7/20 -- none of them 15/20.

So the August numbers were produced with at least one flag that differs from
anything recorded, and/or with `sonic_v1_1` weights other than the ones now on
disk (those `.onnx` files are gitignored, so their provenance cannot be checked
from the repository). **Use the numbers in `results/` and not the August log.**

## Machine and environment

| Item | Value |
|---|---|
| CPU | 13th Gen Intel Core i9-13900K, 24 cores / 32 threads |
| GPU | unused (ONNX Runtime `CPUExecutionProvider` only) |
| OS | Ubuntu 24.04.5 LTS, kernel 6.8.0-146-generic |
| Python | 3.10.20 (`~/miniconda3/envs/mm-g1-sonic`) |
| mujoco | 3.3.3 |
| onnxruntime | 1.23.2 |
| numpy | 2.2.6 |
| matplotlib | 3.10.9 |

The policy is deterministic given the checkpoint and the flags, and MuJoCo is
run without added noise, so repeated runs of a condition reproduce the same
per-clip outcomes on this machine.

## Clip list (`mm_g1.data._box_clip_names()`, 20 stems)

```
sub10_largebox_036_original_mujoco  sub7_largebox_042_original_mujoco
sub10_largebox_081_original_mujoco  sub7_largebox_043_original_mujoco
sub10_largebox_082_original_mujoco  sub7_largebox_044_original_mujoco
sub10_largebox_083_original_mujoco  sub7_largebox_045_original_mujoco
sub10_largebox_084_original_mujoco  sub7_largebox_046_original_mujoco
sub10_largebox_085_original_mujoco  sub7_largebox_047_original_mujoco
sub10_largebox_087_original_mujoco  sub8_largebox_042_original_mujoco
sub10_largebox_088_original_mujoco  sub8_largebox_043_original_mujoco
sub10_largebox_089_original_mujoco
sub12_largebox_071_original_mujoco
sub16_largebox_043_original_mujoco
sub3_largebox_005_original_mujoco
```

The clip `.npz` files live in `data/robot_object_g1/`
(`mm_g1.config.BOX_DATA_DIR`).

## Plot colors

`make_plot.py` uses categorical slots 1 and 2 of the validated reference
palette -- light `#2a78d6` / `#eb6834` on surface `#fcfcfb`, dark `#3987e5` /
`#d95926` on surface `#1a1a19`. Both pairs pass the lightness-band, chroma,
CVD-separation, normal-vision and contrast checks of the data-viz validator, so
the figure is readable in print, in dark mode, and under colour-vision
deficiency. Identity is carried by a legend plus the printed count on every
column, never by hue alone.
