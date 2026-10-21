#!/usr/bin/env python3
"""Squeeze-test sweep: drive sonic_g1_box/run_track_clips.py as subprocesses.

One subprocess per (condition, SONIC checkpoint); 20 clips inside each.
Writes eval_v02/results/squeeze_results.{json,csv} and per-run logs.

    MUJOCO_GL=egl python eval_v02/run_squeeze_sweep.py
"""
import argparse
import concurrent.futures as cf
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, 'results')
LOGS = os.path.join(RESULTS, 'logs')
TRACKER = os.path.join(ROOT, 'sonic_g1_box', 'run_track_clips.py')

CHECKPOINTS = ('release', 'sonic_v1_1')

# Every condition also carries the tracker defaults --shoulder-open 0.4,
# --box-friction 1.5, --box-offset 0.05, --substeps 20, --settle 1.0,
# --speed 1.0; --box-mass is pinned explicitly below.
COMMON = ['--box-mass', '0.25']

CONDITIONS = [
    dict(key='none', label='no squeeze',
         args=['--shoulder-squeeze', '0', '--wrist-squeeze', '0',
               '--palm-force', '0']),
    dict(key='shoulder', label='shoulder 0.6 rad',
         args=['--shoulder-squeeze', '0.6', '--wrist-squeeze', '0',
               '--palm-force', '0']),
    dict(key='wrist', label='wrist 0.2 rad',
         args=['--shoulder-squeeze', '0', '--wrist-squeeze', '0.2',
               '--palm-force', '0']),
    dict(key='shoulder+wrist', label='shoulder 0.6 + wrist 0.2',
         args=['--shoulder-squeeze', '0.6', '--wrist-squeeze', '0.2',
               '--palm-force', '0']),
    dict(key='palm15', label='palm 15 N',
         args=['--shoulder-squeeze', '0', '--wrist-squeeze', '0',
               '--palm-force', '15']),
    dict(key='palm25', label='palm 25 N',
         args=['--shoulder-squeeze', '0', '--wrist-squeeze', '0',
               '--palm-force', '25']),
    dict(key='palm40', label='palm 40 N',
         args=['--shoulder-squeeze', '0', '--wrist-squeeze', '0',
               '--palm-force', '40']),
    # run_track_clips.py adds the label-gated shoulder/wrist biases whenever
    # they are non-zero, independent of --palm-force, so the tracker's own
    # defaults mean "palm force on top of both biases".  This is the code
    # path behind the August sweep log out/track_clips/sweep_palmforce25.log.
    dict(key='palm25+biases', label='palm 25 N + shoulder 0.6 + wrist 0.2',
         args=['--shoulder-squeeze', '0.6', '--wrist-squeeze', '0.2',
               '--palm-force', '25']),
]

LINE = re.compile(
    r'^\s+(\S+)\s+(SUCCESS|slipped|dropped|FELL@(-?\d+))\s+'
    r'peak\s+(-?[\d.]+)\s+\(ref\s+(-?[\d.]+)\)\s+end z\s+(-?[\d.]+)\s*$')
SUMMARY = re.compile(r'->\s+(\d+)/(\d+)\s+carried without dropping')
STATUS = {'SUCCESS': 'success', 'slipped': 'slipped', 'dropped': 'dropped'}

DEFINITIONS = {
    'task': 'Kinematic-reference tracking of 20 recorded OmniRetarget G1 '
            'large-box carry clips with the SONIC tracker in MuJoCo physics '
            'and a frictional carton. Not the full pick-carry-place pipeline: '
            'the robot and the box start at the clip\'s first frame and the '
            'clip itself is the reference the tracker follows.',
    'success': 'lifted and not fallen and z_at_carry_end > z_rest + 0.15, '
               'where lifted = peak box z > z_rest + 0.25, z_rest = median '
               'box z over the first 5 clip frames, and carry_end is the last '
               '50 Hz frame at which the reference box z exceeds '
               'z_rest + 0.25 (run_track_clips.py).',
    'outcomes': {
        'success': 'carried without dropping (the success predicate holds)',
        'dropped': 'box lifted above z_rest + 0.25 but not still held at '
                   'carry_end',
        'slipped': 'box never lifted above z_rest + 0.25',
        'fell': 'robot pelvis z fell below FALL_Z = 0.28 m; the episode is '
                'cut at that frame',
    },
    'fall_threshold_m': 0.28,
    'settle_s': 1.0,
    'substeps': 20,
    'control_dt_s': 0.02,
    'physics_timestep_s': 0.001,
    'playback_speed': 1.0,
    'box_mass_kg': 0.25,
    'box_friction': 1.5,
    'box_scale': 1.0,
    'box_offset_m': 0.05,
    'box_type': 'carton',
    'shoulder_open_rad': 0.4,
    'clip_list_source': 'mm_g1.data._box_clip_names()',
    'tracker_script': 'sonic_g1_box/run_track_clips.py',
}


def parse(text, n_expected):
    outcomes, peaks, refs, ends, falls = {}, {}, {}, {}, {}
    for line in text.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        stem, raw, fall_f, peak, ref, end = m.groups()
        outcomes[stem] = STATUS.get(raw, 'fell')
        if raw not in STATUS:
            falls[stem] = int(fall_f)
        peaks[stem] = float(peak)
        refs[stem] = float(ref)
        ends[stem] = float(end)
    carried = sum(v == 'success' for v in outcomes.values())
    reported = None
    m = SUMMARY.search(text)
    if m:
        reported = (int(m.group(1)), int(m.group(2)))
        if reported[0] != carried:
            raise RuntimeError(f'parsed {carried} successes but the tracker '
                               f'reported {reported[0]}')
    if len(outcomes) != n_expected:
        raise RuntimeError(f'parsed {len(outcomes)} clip lines, '
                           f'expected {n_expected}')
    return outcomes, peaks, refs, ends, falls


def run_one(cond, ckpt, clips, python):
    cmd = [python, TRACKER, '--sonic', ckpt] + COMMON + cond['args']
    if clips:
        cmd += ['--clips'] + clips
    env = dict(os.environ, MUJOCO_GL='egl', CUDA_VISIBLE_DEVICES='')
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    log = os.path.join(LOGS, f'{cond["key"].replace("+", "_plus_")}__{ckpt}.log')
    with open(log, 'w') as fh:
        fh.write('# ' + ' '.join(cmd) + '\n')
        fh.write(f'# cwd {ROOT}\n# MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=\n')
        fh.write(f'# exit {proc.returncode} wall {wall:.1f}s\n\n')
        fh.write(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f'{cond["key"]}/{ckpt} exited {proc.returncode}; '
                           f'see {log}')
    return cmd, wall, log, proc.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--clips', nargs='*', default=None,
                    help='restrict to these clip stems (smoke tests)')
    ap.add_argument('--conditions', nargs='*', default=None)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--out', default=os.path.join(RESULTS,
                                                  'squeeze_results.json'))
    args = ap.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    sys.path.insert(0, ROOT)
    from mm_g1.data import _box_clip_names
    clips = args.clips or list(_box_clip_names())
    conds = [c for c in CONDITIONS
             if args.conditions is None or c['key'] in args.conditions]

    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE).stdout.strip()
    jobs = [(c, k) for c in conds for k in CHECKPOINTS]
    print(f'{len(jobs)} runs x {len(clips)} clips, {args.jobs} in parallel')

    rows, t_start = {}, time.time()
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        fut = {pool.submit(run_one, c, k, args.clips, args.python): (c, k)
               for c, k in jobs}
        for f in cf.as_completed(fut):
            cond, ckpt = fut[f]
            cmd, wall, log, out = f.result()
            outcomes, peaks, refs, ends, falls = parse(out, len(clips))
            carried = sum(v == 'success' for v in outcomes.values())
            rows[(cond['key'], ckpt)] = dict(
                key=cond['key'], label=cond['label'],
                args=COMMON + cond['args'], checkpoint=ckpt,
                command=' '.join(cmd), n_clips=len(clips), carried=carried,
                outcomes=outcomes, peak_lift_m=peaks, ref_peak_m=refs,
                end_z_m=ends, fall_frame=falls,
                wall_seconds=round(wall, 1),
                seconds_per_clip=round(wall / len(clips), 2),
                log=os.path.relpath(log, ROOT))
            print(f'  {cond["key"]:16s} {ckpt:11s} {carried:2d}/{len(clips)} '
                  f'({wall:.0f}s)')

    conditions = [rows[(c['key'], k)] for c in conds for k in CHECKPOINTS]
    doc = dict(
        generated=datetime.datetime.now().astimezone().isoformat(
            timespec='seconds'),
        git_commit=commit, clips=clips, checkpoints=list(CHECKPOINTS),
        sweep_wall_seconds=round(time.time() - t_start, 1),
        conditions=conditions, definitions=DEFINITIONS)
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(doc, fh, indent=2)

    csv_path = os.path.splitext(args.out)[0] + '.csv'
    cols = ['key', 'label', 'checkpoint', 'n_clips', 'carried', 'success',
            'slipped', 'dropped', 'fell', 'wall_seconds', 'seconds_per_clip',
            'args']
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for c in conditions:
            tally = {k: 0 for k in ('success', 'slipped', 'dropped', 'fell')}
            for v in c['outcomes'].values():
                tally[v] += 1
            w.writerow([c['key'], c['label'], c['checkpoint'], c['n_clips'],
                        c['carried'], tally['success'], tally['slipped'],
                        tally['dropped'], tally['fell'], c['wall_seconds'],
                        c['seconds_per_clip'], ' '.join(c['args'])])
    print(f'wrote {args.out}\nwrote {csv_path}')


if __name__ == '__main__':
    main()
