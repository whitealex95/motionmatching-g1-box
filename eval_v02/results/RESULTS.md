# Squeeze test: clips carried out of 20

Source: `eval_v02/results/squeeze_results.json`  
Sweep generated: 2026-10-21T00:18:20-04:00  
Repo commit at sweep time: `54318c8556f9c8464a7f06f26a180cb38c4a67e1`  
Tables generated: 2026-10-21T00:30:46-04:00

Kinematic-reference tracking of the 20 OmniRetarget G1 large-box carry clips with the SONIC tracker in MuJoCo physics and a frictional carton. Not the full pick-carry-place pipeline. `carried` counts clips meeting the success predicate; `slip` = never lifted, `drop` = lifted but not still held at the reference put-down, `fell` = robot pelvis below 0.28 m.

| Strategy | release carried | sonic_v1_1 carried | release slip/drop/fell | sonic_v1_1 slip/drop/fell |
|---|---|---|---|---|
| no squeeze (`none`) | **0**/20 | **0**/20 | 20/0/0 | 19/0/1 |
| shoulder 0.6 rad (`shoulder`) | **0**/20 | **0**/20 | 13/7/0 | 14/5/1 |
| wrist 0.2 rad (`wrist`) | **0**/20 | **0**/20 | 18/2/0 | 16/3/1 |
| shoulder 0.6 + wrist 0.2 (`shoulder+wrist`) | **8**/20 | **0**/20 | 9/3/0 | 9/10/1 |
| palm 15 N (`palm15`) | **10**/20 | **9**/20 | 10/0/0 | 8/2/1 |
| palm 25 N (`palm25`) | **12**/20 | **12**/20 | 7/1/0 | 6/1/1 |
| palm 40 N (`palm40`) | **12**/20 | **13**/20 | 6/2/0 | 5/1/1 |
| palm 25 N + shoulder 0.6 + wrist 0.2 (`palm25+biases`) | **13**/20 | **6**/20 | 6/1/0 | 5/8/1 |

## Per-clip outcomes

### release

| Clip | `none` | `shoulder` | `wrist` | `shoulder+wrist` | `palm15` | `palm25` | `palm40` | `palm25+biases` |
|---|---|---|---|---|---|---|---|---|
| sub10_largebox_036 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |
| sub10_largebox_081 | slipped | slipped | slipped | success | success | success | success | success |
| sub10_largebox_082 | slipped | slipped | slipped | slipped | slipped | dropped | dropped | success |
| sub10_largebox_083 | slipped | slipped | slipped | success | success | success | success | success |
| sub10_largebox_084 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |
| sub10_largebox_085 | slipped | slipped | slipped | success | slipped | success | success | success |
| sub10_largebox_087 | slipped | dropped | slipped | success | success | success | success | success |
| sub10_largebox_088 | slipped | dropped | slipped | success | success | success | success | success |
| sub10_largebox_089 | slipped | slipped | slipped | slipped | success | success | success | slipped |
| sub12_largebox_071 | slipped | dropped | slipped | dropped | success | success | success | dropped |
| sub16_largebox_043 | slipped | slipped | slipped | slipped | slipped | success | success | success |
| sub3_largebox_005 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |
| sub7_largebox_042 | slipped | dropped | dropped | dropped | success | success | success | success |
| sub7_largebox_043 | slipped | dropped | dropped | success | success | success | success | success |
| sub7_largebox_044 | slipped | dropped | slipped | success | success | success | success | success |
| sub7_largebox_045 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |
| sub7_largebox_046 | slipped | dropped | slipped | success | success | success | success | success |
| sub7_largebox_047 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |
| sub8_largebox_042 | slipped | slipped | slipped | slipped | slipped | slipped | dropped | success |
| sub8_largebox_043 | slipped | slipped | slipped | dropped | slipped | slipped | slipped | success |

### sonic_v1_1

| Clip | `none` | `shoulder` | `wrist` | `shoulder+wrist` | `palm15` | `palm25` | `palm40` | `palm25+biases` |
|---|---|---|---|---|---|---|---|---|
| sub10_largebox_036 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |
| sub10_largebox_081 | slipped | slipped | slipped | dropped | success | success | success | dropped |
| sub10_largebox_082 | slipped | slipped | slipped | slipped | slipped | success | success | dropped |
| sub10_largebox_083 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |
| sub10_largebox_084 | slipped | slipped | slipped | slipped | dropped | success | success | success |
| sub10_largebox_085 | slipped | slipped | slipped | dropped | success | success | success | dropped |
| sub10_largebox_087 | slipped | dropped | slipped | dropped | success | success | success | dropped |
| sub10_largebox_088 | slipped | slipped | slipped | dropped | success | success | success | dropped |
| sub10_largebox_089 | fell | fell | fell | fell | fell | fell | fell | fell |
| sub12_largebox_071 | slipped | slipped | dropped | dropped | dropped | dropped | dropped | dropped |
| sub16_largebox_043 | slipped | slipped | slipped | slipped | slipped | success | success | success |
| sub3_largebox_005 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |
| sub7_largebox_042 | slipped | dropped | dropped | dropped | success | success | success | dropped |
| sub7_largebox_043 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |
| sub7_largebox_044 | slipped | slipped | slipped | dropped | success | success | success | success |
| sub7_largebox_045 | slipped | dropped | slipped | dropped | success | success | success | dropped |
| sub7_largebox_046 | slipped | dropped | dropped | dropped | success | success | success | success |
| sub7_largebox_047 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |
| sub8_largebox_042 | slipped | slipped | slipped | slipped | slipped | slipped | success | success |
| sub8_largebox_043 | slipped | dropped | slipped | dropped | success | success | success | success |

