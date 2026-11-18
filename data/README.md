# Motion data

The databases that `mm_g1` (the box task's controller, `sonic_g1_box/mm_driver.py`) reads. They are copies of the files that
lmm-g1-box's page loads for its default variant `holosoma-v3-box338x335x360-apv5-good2-sbplace-angle60` (`sbplace`)
with the Locomotion Matching Database `emmboth`, the page's default. Nothing here is regenerated: the files are
copied byte for byte from lmm-g1-box `121379f` (v4.2.1, 2026-11-18), so this repository runs without lmm-g1-box. Since
`51f3eae` (v4.0) two things changed. v4.2 took `nlatent`, the latent size of the Learned Motion Matching networks that
the page removed, out of `sbplace/config.json`. v4.2.1 left the EMM takes `strafe` and `turn_in_place` out of
`emm_loco/`: 14 takes stay, 106,584 frames with their mirrors (119,996 before). The retargeting of the two has the
pelvis about 19 deg back and the waist pitch about 27 deg forward in every frame, and SONIC release holds the waist
pitch near 0 and follows the pelvis, so on them the robot stood, strafed and turned with its torso leaning back.
Strafing and turning in place now play from the other takes. Before
2026-11-16 `sbplace/` held the variant `holosoma-v3-box338x335x360-apv4-scenebox-sbplace-angle60` (lmm-g1-box
`a41bd3f`, ApproachPick v4 with the SceneBot pick). `emm_carry/` and `robot.*` are the same in both.

| here | lmm-g1-box `web/data/` | what it holds |
|---|---|---|
| `sbplace/config.json` | `holosoma-v3-box338x335x360-apv5-good2-sbplace-angle60/config.json` | the variant: clips, ranges, feature layout, box size, controller settings |
| `sbplace/mm_box.bin` | `holosoma-v3-box338x335x360-apv5-good2-sbplace-angle60/mm_box.bin` | the Box Matching Database: 12,832 frames in 62 ranges, the picks of the ApproachPick v5 `xy_yaw` clips (the pick indexed once) and one place |
| `sbplace/mm_pose.bin` | `holosoma-v3-box338x335x360-apv5-good2-angle60/mm_pose.bin` | the Animation Database the variant shares with `apv5-good2-angle60` (`shares` in lmm-g1-box's `lmm/variants.py`) |
| `sbplace/mm_loco.bin` | `holosoma-v3-box338x335x360-apv5-good2-angle60/mm_loco.bin` | the variant's own Locomotion Matching Database (`--loco nsm`), shared the same way |
| `physics/scene.xml`, `physics/sonic.json` | `physics/scene.xml`, `physics/sonic.json` | the simulated G1 of the page's physical mode (GR00T-WholeBodyControl's), which the pick motion's whole-body IK solves on |
| `emm_loco/` | `emm_loco/holosoma-v3/` | 14 of the 16 EMM locomotion takes retargeted with holosoma v3 (`strafe` and `turn_in_place` left out since v4.2.1): Locomotion Matching Database and its poses |
| `emm_carry/` | `emm_carry/` | the EMM carry spans, the same way |
| `robot.json` | `robot.json` | the G1's bodies and joints for forward kinematics and the foot locking |
| `robot.bin` | `robot.bin` | the G1's meshes, which the database page (`web/`) draws |
| `HOLOSOMA-LICENSE.txt`, `G1-LICENSE.txt` | the same names | the licenses below |
| `physics/G1-SONIC-LICENSE.txt` | `assets/g1_sonic/LICENSE` | the license of the simulated G1 |

`emmboth` is built at load time from `emm_carry/` (taken while the robot holds the box) and `emm_loco/` (with empty
hands), as the page builds it.

## SHA-256

```
bd837fb80b551af5f067bbd266d77b959523ca044b31eebc449ef2e16761250d  sbplace/config.json
1e9fcc2d7a802dd0e0412529518f0b421704ce557d98d6d3c72c0382159e0e2a  sbplace/mm_box.bin
b15661ed3c7a2e02c57adfa8de892ebab14043573a5962a1d425598461105d4f  sbplace/mm_loco.bin
065ad6ead90cca2e5ff338875abe86987c7085dfefa3263fd0b20507d2e9f16a  sbplace/mm_pose.bin
04b246b6d9cc724a11d3d7bdff9b189508f54223c61985f26a490036f0f6a0cc  physics/scene.xml
ea2d0abeaec9530bf0fb4f83db7ae24116723b19a44ed779880c8d2a77fbb745  physics/sonic.json
24ab66be50d1aca4fc5e029ef76ce4ceaac6557ea21665caf4b140695a76ffee  physics/G1-SONIC-LICENSE.txt
579c0cc361143d66de7886c0328095940035c750ecfec18e0a41bf57123a79f6  emm_loco/config.json
7a17d3ec85b96f0124ed5cff824783c2fcd7da5e6cc054f0b55f9778934e356a  emm_loco/mm_loco.bin
63a056d545fa64422ade19544a906c5d1402fd4cef761d7153fea3e906f64909  emm_loco/mm_pose.bin
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
* The picks (`sbplace/mm_box.bin`): the ApproachPick v5 clips of holosoma-jh (branch `approach-pick-v5`, reassembled
  at `8f2fc861`). Their walks are Kimodo-G1 walks timed as the page's controller walks to the stance, and each ends in
  the pick `good2-ori` of lmm-g1-box's pick editor, an edit of the pick of clip 11 of the SceneBot web demo
  ([ericcsr.github.io/scenebot](https://ericcsr.github.io/scenebot/)), S. Chen, S. Zhao, Z. Wu, J. Li et al.,
  *SceneBot: Contact-Prompted General Humanoid Whole Body Tracking with Scene-Interaction*. The place is the same pick
  played backward. The SceneBot demo files have no stated license, so these frames are not covered by the licenses
  here.
* `sbplace/mm_pose.bin` and `sbplace/mm_loco.bin` also hold the Neural State Machine episodes of the variant
  (Starke et al. 2019, CC BY-NC 4.0, non-commercial use only).
* Retargeting: holosoma (Apache 2.0, `HOLOSOMA-LICENSE.txt`).
* Robot model (`robot.json`, `robot.bin`): the Unitree G1 description, BSD 3-Clause (`G1-LICENSE.txt`).
* The simulated G1 (`physics/scene.xml`, `physics/sonic.json`): from
  [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), Apache 2.0 for the code and the
  model files (`physics/G1-SONIC-LICENSE.txt`, which also carries the NVIDIA Open Model License of SONIC's weights,
  none of which are here).
* Method: [orangeduck/Motion-Matching](https://github.com/orangeduck/Motion-Matching) (MIT), as ported in
  lmm-g1-box's `web/js/` and from there to `mm_g1/`.
