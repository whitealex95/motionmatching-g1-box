#!/usr/bin/env python3
"""Validate the complete 720-video sweep and its unchanged baseline episodes."""
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/box_all_videos'
ORDER = ['release', 'sonic_v1_1', 'supermimic', 'supermimic_before_force']
NAMES = ['SONIC release', 'SONIC v1.1', 'SuperMimic, clip after force',
         'SuperMimic, clip before force']
FIELDS = ['success', 'fallen', 'peak', 'carry_end_z', 'end_z', 'fall_f']


def check_video(episode):
    path = Path(episode['video'])
    metadata = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,avg_frame_rate,nb_frames,duration,codec_name',
        '-of', 'json', str(path)], text=True))['streams'][0]
    assert metadata['width'] == 1280 and metadata['height'] == 720, path
    assert metadata['avg_frame_rate'] == '25/1' and metadata['codec_name'] == 'h264', path
    assert int(metadata['nb_frames']) > 25 and float(metadata['duration']) > 1, path
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **metadata)


def main():
    doc = json.loads((OUT / 'comparison.json').read_text())
    old = json.loads((HERE / 'results/box/comparison.json').read_text())
    keys = [c['key'] for c in doc['definitions']['conditions']]
    rows = {(r['key'], r['checkpoint']): r for r in doc['conditions']}
    assert doc['complete'] and len(doc['clips']) == 20 and len(keys) == 9
    assert set(rows) == {(k, t) for k in keys for t in ORDER}
    episodes = []
    for (key, tracker), row in rows.items():
        assert row['n_clips'] == len(row['episodes']) == 20
        assert set(row['outcomes']) == {e['clip'] for e in row['episodes']} == set(doc['clips'])
        assert row['carried'] == sum(e['success'] for e in row['episodes'])
        for e in row['episodes']:
            expected = (e['peak'] > e['z_rest']+.25 and not e['fallen']
                        and e['carry_end_z'] is not None and e['carry_end_z'] > e['z_rest']+.15)
            assert e['success'] == expected, (key, tracker, e['clip'])
            if tracker == 'supermimic':
                assert e['torque']['max_limit_excess_nm'] < 1e-6
            if tracker == 'supermimic_before_force':
                assert not any(e['actuator_control_limited'])
                assert not any(e['actuator_force_limited'])
                assert not any(e['joint_actuator_force_limited'])
            episodes.append(e)
    assert len({e['video'] for e in episodes}) == 720
    baseline_count = 0
    for row in old['conditions']:
        current = {e['clip']: e for e in rows[row['key'], row['checkpoint']]['episodes']}
        for e in row['episodes']:
            for field in FIELDS:
                assert current[e['clip']][field] == e[field], (row['key'], row['checkpoint'], e['clip'], field)
            baseline_count += 1
    no_force_count = 0
    for key in ['none', 'shoulder', 'wrist', 'shoulder+wrist']:
        after = {e['clip']: e for e in rows[key, 'supermimic']['episodes']}
        for e in rows[key, 'supermimic_before_force']['episodes']:
            for field in FIELDS:
                assert e[field] == after[e['clip']][field], (key, e['clip'], field)
            no_force_count += 1
    with ThreadPoolExecutor(max_workers=6) as pool:
        videos = list(pool.map(check_video, episodes))
    (OUT / 'video_manifest.json').write_text(json.dumps(videos, indent=2)+'\n')
    force_rows = [r for r in rows.values() if r['checkpoint'] == 'supermimic_before_force']
    exceeding = sum(e['torque']['substeps_exceeding_limits'] > 0 for r in force_rows for e in r['episodes'])
    assert exceeding > 0
    validation = dict(episodes=720, videos=len(videos), exact_baseline_repeats=baseline_count,
        exact_no_virtual_force_repeats=no_force_count, before_force_episodes_exceeding_demo_limits=exceeding,
        max_before_force_excess_nm=max(e['torque']['max_limit_excess_nm'] for r in force_rows for e in r['episodes']),
        video_bytes=sum(v['bytes'] for v in videos), passed=True)
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    with (OUT / 'summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['controller', *keys])
        for tracker, name in zip(ORDER, NAMES):
            writer.writerow([name, *[rows[key, tracker]['carried'] for key in keys]])
    lines = ['# All OmniRetarget box recordings', '',
        '720 measured and recorded episodes: 20 clips × 9 heuristics × 4 controller settings.',
        'Each cell below reports successful carries out of 20.', '',
        '| Controller | '+' | '.join(keys)+' |',
        '|---|'+'---:|'*len(keys)]
    for tracker, name in zip(ORDER, NAMES):
        lines.append('| '+name+' | '+' | '.join(str(rows[k, tracker]['carried']) for k in keys)+' |')
    lines += ['',
        'Palm force is specified per labelled palm. Each virtual force points toward the box centre, ramps over 0.15 seconds, and is converted to arm torque with the palm Jacobian.',
        'The new SuperMimic row clips PD torque to the existing demo and scene limits before adding virtual-force torque. MuJoCo motor control, actuator force, and joint actuator-force clips are disabled for this row. The resulting applied torque can exceed these limits. The original SuperMimic and SONIC rows retain their existing torque paths.',
        'Success requires peak box height above resting height + 0.25 m, no robot fall, and box height above resting height + 0.15 m at the reference carry end. The carton mass is 0.25 kg, friction is 1.5, and physical offset is 0 m.',
        'Videos are 1280 × 720 H.264 at 25 fps. Policy control remains 50 Hz, with 20 physics substeps per control tick. Each recording includes one second of settling. Runs stop if the robot falls.', '',
        f'All {baseline_count} original baseline episodes exactly reproduce their stored outcomes, peak height, carry-end height, final height, and fall frame.',
        f'All {no_force_count} SuperMimic episodes without virtual force exactly match between clipping orders.',
        f'{exceeding} new-row episodes exert joint actuator torque above the demo limit. The largest measured excess is {validation["max_before_force_excess_nm"]:.6f} N·m. Diagnostics use MuJoCo qfrc_actuator at every physics substep, including settling.',
        'All 720 video files passed container, frame count, resolution, codec, and frame-rate checks. video_manifest.json records file sizes, durations, frame counts, and SHA-256 hashes.',
        'comparison.json records commands, dependencies, source and asset hashes, per-episode values, and wall time. No training was performed.', '']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
