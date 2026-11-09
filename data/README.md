# Motion data

The databases that `mm_g1` (the box task's controller, `sonic_g1_box/mm_driver.py`) reads. They are copies of the files that
lmm-g1-box's page loads for the variant `holosoma-v3-box338x335x360-apv4-scenebox-sbplace-angle60` (`sbplace`) with
the Locomotion Matching Database `emmboth`, the page's default. Nothing here is regenerated: the files are copied
byte for byte from lmm-g1-box commit `a41bd3f` (2026-11-04), so this repository runs without lmm-g1-box. They are
the same in `1721681`, the commit `mm_g1` is ported from, and in the tag `v3.0` (`6cfb32d`), which made this variant
and `emmboth` the page's defaults. `robot.bin` is from `v3.0`.

| here | lmm-g1-box `web/data/` | what it holds |
|---|---|---|
| `sbplace/config.json` | `holosoma-v3-box338x335x360-apv4-scenebox-sbplace-angle60/config.json` | the variant: clips, ranges, feature layout, box size, controller settings |
| `sbplace/mm_box.bin` | `holosoma-v3-box338x335x360-apv4-scenebox-sbplace-angle60/mm_box.bin` | the Box Matching Database: 12,737 frames in 67 ranges, the picks of the ApproachPick v4 `xy_yaw` clips and one SceneBot place |
| `sbplace/mm_pose.bin` | `holosoma-v3-box338x335x360-apv4-scenebox-angle60/mm_pose.bin` | the Animation Database the variant shares with `scenebox` (`shares` in lmm-g1-box's `lmm/variants.py`) |
| `sbplace/mm_loco.bin` | `holosoma-v3-box338x335x360-apv4-scenebox-angle60/mm_loco.bin` | the variant's own Locomotion Matching Database (`--loco nsm`), shared the same way |
| `emm_loco/` | `emm_loco/holosoma-v3/` | the EMM locomotion takes retargeted with holosoma v3: Locomotion Matching Database and its poses |
| `emm_carry/` | `emm_carry/` | the EMM carry spans, the same way |
| `robot.json` | `robot.json` | the G1's bodies and joints for forward kinematics and the foot locking |
| `robot.bin` | `robot.bin` | the G1's meshes, which the database page (`web/`) draws |
| `HOLOSOMA-LICENSE.txt`, `G1-LICENSE.txt` | the same names | the licenses below |

`emmboth` is built at load time from `emm_carry/` (taken while the robot holds the box) and `emm_loco/` (with empty
hands), as the page builds it.

## SHA-256

```
cf8f0ac96b97f1d205ecac39400c2741026c92999798a7b31652e9bacb913c57  sbplace/config.json
167cb7d6ca0931ee093546726b0ebb304cf43943489c057865dc8a067e8a557f  sbplace/mm_box.bin
b03a2e96064be7320a1315052907279c3cee6c098b8869951dcaa5a3df909ea2  sbplace/mm_loco.bin
475a4b098a3e2735c9ddf87f445901931795743b0a3f8d85ede48ffc8c21c61c  sbplace/mm_pose.bin
205b48451247bd70339668293440ab812c32a5a3cf452228acfd765847841f89  emm_loco/config.json
5eb0b3fa516b0b8996856a04fd96153c8a6fa03b2d768d0e9d9c110c6c1a436e  emm_loco/mm_loco.bin
896a59535ae16bfef7fabfc789b1066338b149daa6168f31cb62385369e4edcc  emm_loco/mm_pose.bin
62ed3f56c5dde207976cd0fd411367cd04c451b96689da46c8aa693f628073ad  emm_carry/config.json
480b0af91ffc75d9a6e33115f2e443cceb094d83a2c805729172b4b8ee95fcb0  emm_carry/mm_loco.bin
4667537e84f730e8f0b2356835f6724175a8ad05b1018f9bbf7dcadde35984a1  emm_carry/mm_pose.bin
a80dc63a59a859e92a3a288420bdeada19fb52fbf18277d99e7db6978ef2bec9  robot.json
6fc851dd97ec61b3cb4c6f10cb44ee4b5a822be868caba039103abdda7fba2ed  robot.bin
09e8a9bcec8067104652c168685ab0931e7868f9c8284b66f5ae6edae5f1130b  HOLOSOMA-LICENSE.txt
84aac59fd3246e3ddc49d1387644e8fcf43b0def4f0b9fe687f372e90446df2d  G1-LICENSE.txt
```

`cd data && sha256sum -c` with these lines checks a copy.

## Where the motions come from, and their terms

* Locomotion (`emm_loco/`, `emm_carry/`): the locomotion takes of J. L. Ponton, S. Andrews, C. Andújar, N. Pelechano,
  *Environment-aware Motion Matching*, ACM TOG (SIGGRAPH Asia 2025), data on
  [Zenodo](https://zenodo.org/records/17220516), CC BY 4.0. Retargeted to the G1.
* The picks (`sbplace/mm_box.bin`): the ApproachPick v4 clips of holosoma-jh (made on 2026-11-02 at holosoma-jh commit
  `1f08cba`). Their walks are Kimodo-G1 walks, and each ends in the pick of clip 11 of the SceneBot web demo
  ([ericcsr.github.io/scenebot](https://ericcsr.github.io/scenebot/)), S. Chen, S. Zhao, Z. Wu, J. Li et al.,
  *SceneBot: Contact-Prompted General Humanoid Whole Body Tracking with Scene-Interaction*. The place is the same clip
  played backward. The SceneBot demo files have no stated license, so these frames are not covered by the licenses
  here.
* `sbplace/mm_pose.bin` and `sbplace/mm_loco.bin` also hold the Neural State Machine episodes of the variant
  (Starke et al. 2019, CC BY-NC 4.0, non-commercial use only).
* Retargeting: holosoma (Apache 2.0, `HOLOSOMA-LICENSE.txt`).
* Robot model (`robot.json`, `robot.bin`): the Unitree G1 description, BSD 3-Clause (`G1-LICENSE.txt`).
* Method: [orangeduck/Motion-Matching](https://github.com/orangeduck/Motion-Matching) (MIT), as ported in
  lmm-g1-box's `web/js/` and from there to `mm_g1/`.
