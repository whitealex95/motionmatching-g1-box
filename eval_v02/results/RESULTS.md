# Squeeze test: clips carried out of 20

Source: `eval_v02/results/squeeze_results.json`  
Sweep generated: 2026-10-21T20:33:29-04:00  
Repo commit at sweep time: `ab39b50a82a873819103cf4f594f4dc6ab655b92`  
Box offset along the heading: 0.0 m (0 = the August 2026 setting; the tracker script defaults to 0.05)  
Tables generated: 2026-10-21T20:34:00-04:00

Kinematic-reference tracking of the 20 OmniRetarget G1 large-box carry clips with the SONIC tracker in MuJoCo physics and a frictional carton. Not the full pick-carry-place pipeline. `carried` counts clips meeting the success predicate; `slip` = never lifted, `drop` = lifted but not still held at the reference put-down, `fell` = robot pelvis below 0.28 m.

| Strategy | release carried | sonic_v1_1 carried | release slip/drop/fell | sonic_v1_1 slip/drop/fell |
|---|---|---|---|---|
| no squeeze (`none`) | **0**/20 | **0**/20 | 20/0/0 | 19/0/1 |
| shoulder 0.6 rad (`shoulder`) | **0**/20 | **2**/20 | 11/9/0 | 11/6/1 |
| wrist 0.2 rad (`wrist`) | **1**/20 | **0**/20 | 18/1/0 | 14/5/1 |
| shoulder 0.6 + wrist 0.2 (`shoulder+wrist`) | **7**/20 | **1**/20 | 8/5/0 | 7/11/1 |
| palm 15 N (`palm15`) | **11**/20 | **10**/20 | 9/0/0 | 6/3/1 |
| palm 25 N (`palm25`) | **13**/20 | **15**/20 | 7/0/0 | 3/1/1 |
| palm 40 N (`palm40`) | **14**/20 | **16**/20 | 5/1/0 | 3/0/1 |
| palm 25 N + shoulder 0.6 + wrist 0.2 (`palm25+biases`) | **14**/20 | **6**/20 | 5/1/0 | 3/10/1 |

## Per-clip outcomes

### release

| Clip | `none` | `shoulder` | `wrist` | `shoulder+wrist` | `palm15` | `palm25` | `palm40` | `palm25+biases` |
|---|---|---|---|---|---|---|---|---|
| sub10_largebox_036 | slipped | dropped | slipped | dropped | success | success | success | success |
| sub10_largebox_081 | slipped | slipped | slipped | success | slipped | success | success | success |
| sub10_largebox_082 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |
| sub10_largebox_083 | slipped | slipped | slipped | success | success | success | success | success |
| sub10_largebox_084 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |
| sub10_largebox_085 | slipped | slipped | slipped | success | slipped | success | success | success |
| sub10_largebox_087 | slipped | dropped | slipped | success | success | success | success | success |
| sub10_largebox_088 | slipped | dropped | slipped | success | success | success | success | success |
| sub10_largebox_089 | slipped | slipped | slipped | dropped | success | success | success | success |
| sub12_largebox_071 | slipped | dropped | slipped | dropped | success | success | success | success |
| sub16_largebox_043 | slipped | slipped | slipped | slipped | slipped | slipped | success | success |
| sub3_largebox_005 | slipped | dropped | slipped | slipped | success | success | success | dropped |
| sub7_largebox_042 | slipped | dropped | dropped | success | success | success | success | success |
| sub7_largebox_043 | slipped | dropped | success | dropped | success | success | success | success |
| sub7_largebox_044 | slipped | dropped | slipped | dropped | success | success | success | success |
| sub7_largebox_045 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |
| sub7_largebox_046 | slipped | dropped | slipped | success | success | success | success | success |
| sub7_largebox_047 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |
| sub8_largebox_042 | slipped | slipped | slipped | slipped | slipped | slipped | dropped | success |
| sub8_largebox_043 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |

### sonic_v1_1

| Clip | `none` | `shoulder` | `wrist` | `shoulder+wrist` | `palm15` | `palm25` | `palm40` | `palm25+biases` |
|---|---|---|---|---|---|---|---|---|
| sub10_largebox_036 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |
| sub10_largebox_081 | slipped | dropped | dropped | dropped | success | success | success | dropped |
| sub10_largebox_082 | slipped | slipped | slipped | slipped | slipped | dropped | success | dropped |
| sub10_largebox_083 | slipped | slipped | slipped | slipped | dropped | success | success | dropped |
| sub10_largebox_084 | slipped | slipped | slipped | slipped | dropped | success | success | dropped |
| sub10_largebox_085 | slipped | slipped | slipped | dropped | success | success | success | dropped |
| sub10_largebox_087 | slipped | dropped | slipped | dropped | success | success | success | success |
| sub10_largebox_088 | slipped | slipped | slipped | dropped | success | success | success | dropped |
| sub10_largebox_089 | fell | fell | fell | fell | fell | fell | fell | fell |
| sub12_largebox_071 | slipped | dropped | dropped | dropped | success | success | success | dropped |
| sub16_largebox_043 | slipped | slipped | slipped | dropped | slipped | success | success | success |
| sub3_largebox_005 | slipped | dropped | slipped | dropped | success | success | success | success |
| sub7_largebox_042 | slipped | success | dropped | dropped | success | success | success | dropped |
| sub7_largebox_043 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |
| sub7_largebox_044 | slipped | slipped | slipped | dropped | success | success | success | dropped |
| sub7_largebox_045 | slipped | dropped | dropped | dropped | success | success | success | dropped |
| sub7_largebox_046 | slipped | success | dropped | success | success | success | success | success |
| sub7_largebox_047 | slipped | slipped | slipped | slipped | slipped | slipped | slipped | slipped |
| sub8_largebox_042 | slipped | slipped | slipped | slipped | slipped | success | success | success |
| sub8_largebox_043 | slipped | dropped | slipped | dropped | dropped | success | success | success |

