# Re-retargeting the EMM box carry with OmniRetargeting

`data/emm_g1/clips/emm_extra__box.npz` is the EMM `box.bvh` take retargeted with GMR.
GMR knows nothing about the box the actor carried, so the G1 holds its hands as far
apart as the actor did (0.6 m, a ~0.52 m wide box). The box this project picks is
0.3 x 0.2 x 0.3 m and the SceneBot pick leaves the wrists **0.42 m** apart, so the
GMR carry frames hold an invisible box twice the size of the real one.

`retarget_emm_box.py` redoes the retarget with
[OmniRetargeting](https://github.com/project-instinct/omniretargeting) (the open
re-implementation of OmniRetarget) and **shrinks the carried box** on the way:

1. The BVH joints are converted to Z-up metres and scaled by robot height / actor
   height (1.241 / 1.737 = 0.714).
2. The carried box is rebuilt from the hands every frame: centre at the wrist midpoint,
   one axis along the wrist-to-wrist line, width = wrist separation minus a 4 cm hand
   clearance per side (the Unity scene's prop offsets could not be recovered reliably,
   the hands can). A 26-point lattice on its surface joins the interaction mesh.
3. The same lattice, shrunk to `--wrist-sep` minus the scaled clearance, is what the
   robot-side mesh locks to (`MotionData.target_object_points`, see the patch below).
   The Laplacian coordinates still say "wrists just outside the box faces", so the
   wrists close in on the smaller box and the rest of the body follows the actor.
4. OmniRetargeting only matches the wrist *position*, so on its own the palms end up
   facing anywhere (on the full take they mostly faced away from the box). Two virtual
   targets per hand fix that: a palm-centre point and a fingertip point, defined in the
   robot's wrist link frame (`HAND_TARGETS`) and laid out in the box frame on the human
   side. Matching them aligns the hand frame with the box frame: fingers forward along
   the side face, palm on it (`--no-hand-targets` turns this off).
5. The result is saved in the layout `mm_g1/emm_clips.py` already reads: `qpos` (T, 36)
   in the `g1.xml` joint order, `fps`, plus the target `box_pose` (T, 7) for playback.

The scaling alone does most of the work (0.6 m x 0.714 = 0.43 m); the box term pins the
separation exactly and keeps the hands symmetric about the box.

## Setup

OmniRetargeting needs Python >= 3.10 and numpy 2.x, so it gets its own env. The runner
needs one small upstream patch (`omniretargeting-target-object-points.patch`), which adds
`target_object_points` to `MotionData` / `MotionFrame` and threads it to the solver.

```bash
cd ~/Projects
git clone https://github.com/project-instinct/omniretargeting
cd omniretargeting
git checkout -b emm-box 742f630          # the commit the patch was made against
git am ~/Projects/motionmatching-g1-box/tools/omniretarget_emm/omniretargeting-target-object-points.patch
conda create -n omniretargeting python=3.11 -y
conda activate omniretargeting
pip install -e . imageio imageio-ffmpeg
```

The runner finds the G1 URDF and profile through the installed package, so the clone
can live anywhere. The source BVH is read from
`~/Projects/Environment-aware-Motion-Matching/DataEMM/emm_extra/box.bvh` (`--bvh`).

## Usage

```bash
conda activate omniretargeting
cd ~/Projects/motionmatching-g1-box

# quick check on 4 s of the take (~30 s of solver time), with a preview video
MUJOCO_GL=egl python tools/omniretarget_emm/retarget_emm_box.py \
    --start-sec 20 --end-sec 24 --out /tmp/test.npz --video /tmp/test.mp4

# the full take (~22 min at 0.25 s/frame), straight into the clip folder
MUJOCO_GL=egl python tools/omniretarget_emm/retarget_emm_box.py \
    --out data/emm_g1/clips/emm_extra__box_omni.npz \
    --video data/emm_g1/clip_videos/emm_extra__box_omni.mp4

# side-by-side playback of any two clips (left: first, right: --compare)
MUJOCO_GL=egl python tools/omniretarget_emm/render_clip.py \
    data/emm_g1/clips/emm_extra__box_omni.npz /tmp/cmp.mp4 \
    --compare data/emm_g1/clips/emm_extra__box.npz --wrist-box
```

Knobs: `--wrist-sep` (robot wrist separation to aim for, default 0.42 from the SceneBot
hold), `--hand-clearance` (actor wrist joint to box face, 0.04), `--target-box` (the real
box, x forward, y across the hands, only drawn), `--no-box` (plain retarget, for A/B),
`--no-hand-targets` (drop the palm / fingertip targets), `--penetration-resolver xyz_nudge`
(12x faster and far less foot sliding, but it drops the feet 3.5 cm into the floor because
the stabilizer uses the mapped ankle points rather than the sole; not used yet).

The runner prints, for the result, the wrist separation, the wrist-midpoint error to the
box centre, and the hand-facing cosines (palm normal towards the box, finger axis along
the box's forward edge).

**Experimental: `--hold-offset X Z`.** The actor carries the box high (wrist midpoint
0.20 m forward and 0.18 m above the pelvis) while the SceneBot pick leaves it at
0.265 / 0.06, and during CARRY the box is frozen at the pick's pose in the base frame, so
the hands float above the drawn box. This flag edits the actor so the box (and the hands
on it) sit at the requested offset. On the 4 s test the hands do go there (x 0.285,
z 0.049) but the palm-facing cosines drop from 0.93 / 1.00 to 0.36 / 0.46 and the
wrist-midpoint error grows to 7 cm, so it is off by default.

`mm_g1/config/library.py` points `EMM_CLIPS` at the clip to bake; the hold detector in
`mm_g1/emm_clips.py` is unchanged and finds the carry spans in the new clip the same way.
