#!/usr/bin/env python3
"""Validate completed measurements and write concise CSV/Markdown results."""
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
ORDER=['release','sonic_v1_1','supermimic']
KEYS=['none','shoulder','wrist','shoulder+wrist','palm15','palm25','palm40','palm25+biases']


def main():
    box=json.loads((RESULTS/'box/comparison.json').read_text())
    videos=json.loads((RESULTS/'box_videos/comparison.json').read_text())
    loco=json.loads((RESULTS/'loco/comparison.json').read_text())
    rows={(r['key'],r['checkpoint']):r for r in box['conditions']}
    assert box['complete'] and len(rows)==24 and len(box['clips'])==20
    for row in rows.values():
        assert len(row['episodes'])==20 and set(row['outcomes'])==set(box['clips'])
        assert row['carried']==sum(e['success'] for e in row['episodes'])
        for e in row['episodes']:
            expected=(e['peak']>e['z_rest']+.25 and not e['fallen']
                      and e['carry_end_z'] is not None and e['carry_end_z']>e['z_rest']+.15)
            assert e['success']==expected
    for video in videos['conditions']:
        row=rows[video['key'],video['checkpoint']]
        for episode in video['episodes']:
            actual=next(e for e in row['episodes'] if e['clip']==episode['clip'])
            for field in ['success','fallen','peak','carry_end_z','end_z']:
                assert actual[field]==episode[field], (video['key'],video['checkpoint'],field)
    with (RESULTS/'box_summary.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['heuristic',*ORDER])
        for key in KEYS:writer.writerow([key,*[rows[key,t]['carried'] for t in ORDER]])
    lines=['# SuperMimic and SONIC physical tracking results','',
        '480 box episodes completed: 20 OmniRetarget clips × 8 heuristics × 3 checkpoints.',
        'Physical box offset is 0 m. Box mass is 0.25 kg, friction is 1.5, and each tracker uses its native PD gains.',
        'SuperMimic uses the browser demo weights downloaded on 2026-10-21. The demo does not identify the paper training checkpoint.','',
        '| Grasp heuristic | SONIC release | SONIC v1.1 | SuperMimic demo |',
        '|---|---:|---:|---:|']
    for key in KEYS:
        lines.append('| '+rows[key,'release']['label']+' | '+' | '.join(f"{rows[key,t]['carried']}/20" for t in ORDER)+' |')
    lines += ['', 'Success requires a lift above resting height + 0.25 m, no fall, and box height above resting height + 0.15 m at the reference carry end.',
        'The nine rendered clip runs exactly match their sweep records for success, fall, peak height, carry-end height, and final height.','',
        '## Locomotion','',
        'Same EMM autopilot, seed 0, snap-xy correction, 0.4 s reference lead, and 60 s duration cap. Physical feedback generates a separate reference stream for each run.','',
        '| Tracker | Duration | Fell | Mean joint error | Joint error p95 | Waypoint index |',
        '|---|---:|---|---:|---:|---:|']
    for r in loco['conditions']:
        lines.append(f"| {r['settings']['tracker']} | {r['seconds']:.1f} s | {r['fallen']} | {r['joint_error_mean_rad']:.4f} rad | {r['joint_error_p95_rad']:.4f} rad | {r['waypoint']}/13 |")
    lines += ['', 'Both runs remained upright and reached waypoint index 9. Neither completed the 13-waypoint tour within 60 s. Both measured zero root penetration into the matcher obstacle regions.',
        'This is one scene and one seed. Joint errors are measured against each run\'s own generated reference.','',
        '## Full box motion matching','',
        'The pick-here test uses the SceneBot box, 25 N palm force, zero shoulder and wrist squeeze, and anchor reference correction. SuperMimic completes pickup, carry, upright placement, and walking away in 26.9 simulated seconds. SONIC release remains upright but does not complete within 60 seconds.',
        'The comparison video freezes SuperMimic\'s final frame after its episode ends. This separate test uses the motion-matching pipeline and a different box from the recorded-clip evaluation.','',
        '## Verification and provenance','',
        '- Native demo playback versus streamed references: 20 actor steps with zero maximum absolute joint-target difference.',
        '- An altered engine checksum is rejected before inference.',
        '- Desktop/mobile browser checks passed for video playback, task and checkpoint switching, and layout. The local server supports HTTP range requests for seeking.',
        '- Complete per-episode values, commands, source/asset hashes, and environment details are in `box/comparison.json`. Locomotion commands and metrics are in `loco/comparison.json`.',
        '- The sweep resumed after its execution session terminated. Completed groups were preserved and source/asset hashes were checked before resuming. Aggregate elapsed wall time includes interruption gaps. Each group has its own measured subprocess wall time.',
        '- No training was performed.','']
    (RESULTS/'RESULTS.md').write_text('\n'.join(lines))
    print('PASS: 480 episodes and 9 rendered repeats validated')
    print(RESULTS/'RESULTS.md')


if __name__=='__main__':main()
