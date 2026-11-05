#!/usr/bin/env python3
"""Evaluate and record all 20 box clips with explicit force and torque settings."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_v02.run_squeeze_sweep import CONDITIONS as ORIGINAL_CONDITIONS, DEFINITIONS, parse
from mm_g1.data import _box_clip_names

CONDITIONS = [*ORIGINAL_CONDITIONS]
CONDITIONS.insert(-1, dict(key='palm60', label='palm 60 N',
    args=['--shoulder-squeeze', '0', '--wrist-squeeze', '0', '--palm-force', '60']))
CHECKPOINTS = ['release', 'sonic_v1_1', 'supermimic', 'supermimic_before_force']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance():
    sources = {}
    for directory in ('sonic_g1_box', 'tools/sonic_tracking', 'tools/supermimic_tracking', 'mm_g1', 'eval_supermimic', 'eval_v02'):
        for p in (ROOT / directory).rglob('*'):
            if p.is_file() and p.suffix in ('.py', '.mjs'):
                sources[str(p.relative_to(ROOT))] = sha(p)
    for directory in ('assets/scenebot', 'assets/supermimic/demo', 'assets/sonic/policy', 'data/robot_object_g1'):
        for p in (ROOT / directory).rglob('*'):
            if p.is_file() and p.suffix in ('.onnx', '.yaml', '.xml', '.npz', '.json', '.bin', '.cfg', '.wasm'):
                sources[str(p.relative_to(ROOT))] = sha(p)
    return dict(
        timestamp=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        processor=platform.processor(), python=sys.version,
        cpu=subprocess.check_output(['lscpu'], text=True),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        dependencies={p: importlib.import_module(p).__version__ for p in ('numpy', 'mujoco', 'onnxruntime')},
        node=subprocess.check_output(['node', '--version'], text=True).strip(),
        sha256=sources)


def run_one(cond, tracker, clips, args, output):
    key = cond['key'].replace('+', '_plus_') + '__' + tracker
    details = output / 'episodes' / (key + '.json')
    command = [sys.executable, str(ROOT / 'sonic_g1_box/run_track_clips.py'),
        '--tracker', 'supermimic' if tracker.startswith('supermimic') else 'sonic',
        '--torque-clip', 'before-force' if tracker == 'supermimic_before_force' else 'after-force',
        '--clips', *clips, '--box-mass', '0.25', '--box-offset', str(args.box_offset),
        '--box-friction', '1.5', '--box-scale', '1', '--substeps', '20',
        '--settle', '1', '--speed', '1', '--shoulder-open', '0.4',
        '--arm-kp', '1', '--arm-kd', '1', '--results-json', str(details), *cond['args']]
    if not tracker.startswith('supermimic'):
        command += ['--sonic', tracker]
    if args.video:
        command += ['--video', '--video-fps', str(args.video_fps),
                    '--output-dir', str(output / 'videos' / key)]
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, env=dict(os.environ,
        MUJOCO_GL='egl', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', SONIC_ORT_THREADS='2'),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter()-start
    log = output / 'logs' / (key + '.log')
    log.write_text(json.dumps(command) + f'\nexit={proc.returncode} wall_seconds={wall}\n' + proc.stdout)
    if proc.returncode:
        raise RuntimeError(f'{key} exited {proc.returncode}: {log}')
    outcomes, peaks, refs, ends, falls = parse(proc.stdout, len(clips))
    episodes = json.loads(details.read_text())
    if len(episodes['records']) != len(clips):
        raise RuntimeError(f'Incomplete episode output: {key}')
    return dict(key=cond['key'], label=cond['label'], checkpoint=tracker,
        command=command, n_clips=len(clips), carried=sum(v=='success' for v in outcomes.values()),
        outcomes=outcomes, episodes=episodes['records'], wall_seconds=wall,
        log=str(log.relative_to(ROOT)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--box-offset', type=float, default=0.)
    ap.add_argument('--checkpoints', nargs='+', choices=CHECKPOINTS, default=CHECKPOINTS)
    ap.add_argument('--conditions', nargs='+', choices=[c['key'] for c in CONDITIONS])
    ap.add_argument('--clips', nargs='+')
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--video-fps', type=int, choices=[25, 50], default=25)
    ap.add_argument('--resume', action='store_true', help='reuse verified completed groups from partial.json')
    ap.add_argument('--out', type=Path, default=ROOT / 'eval_supermimic/results/box')
    args = ap.parse_args()
    output = args.out.resolve()
    if (output / 'comparison.json').exists():
        raise SystemExit(f'{output} already contains a run. Choose a new --out to preserve provenance.')
    for directory in ('logs', 'episodes', 'videos'):
        (output / directory).mkdir(parents=True, exist_ok=True)
    clips = args.clips or list(_box_clip_names())
    conds = [c for c in CONDITIONS if not args.conditions or c['key'] in args.conditions]
    definitions = dict(DEFINITIONS, box_offset_m=args.box_offset,
        task='Recorded OmniRetarget box clip tracking in MuJoCo with a frictional carton.',
        tracker_controls='Each tracker uses its own PD gains. SuperMimic clips combined PD and virtual-force torque to demo limits. supermimic_before_force clips PD to demo and scene limits first, then adds virtual-force torque with no final clip and disables the scene motor and joint actuator-force clips. SONIC retains its existing scene motor and joint actuator-force limits.',
        recording=dict(enabled=args.video, fps=args.video_fps if args.video else None,
                       width=1280, height=720, control_hz=50),
        conditions=conds,
        timing='Subprocess wall time including initialization, inference, simulation, output and optional rendering. Concurrent runs share CPU resources. No training.')
    current_provenance = provenance()
    partial = output / 'partial.json'
    if args.resume:
        if not partial.exists():
            raise SystemExit(f'No partial results to resume: {partial}')
        doc = json.loads(partial.read_text())
        if (doc['definitions'] != definitions or doc['clips'] != clips
                or doc['checkpoints'] != args.checkpoints):
            raise SystemExit('Resume protocol differs from the saved run')
        # UI and report changes do not affect recorded-clip inference.
        for name, digest in doc['provenance']['sha256'].items():
            if name.startswith(('eval_',)) or name == 'sonic_g1_box/run_interactive.py':
                continue
            if current_provenance['sha256'].get(name) != digest:
                raise SystemExit(f'Resume source or asset changed: {name}')
        doc.setdefault('resumes', []).append(current_provenance)
    else:
        if partial.exists():
            raise SystemExit(f'{partial} exists. Use --resume or choose a new --out.')
        doc = dict(provenance=current_provenance, definitions=definitions, clips=clips,
                   checkpoints=args.checkpoints, conditions=[], jobs=args.jobs, complete=False)
        (output / 'provenance.json').write_text(json.dumps(doc, indent=2))
        partial.write_text(json.dumps(doc, indent=2))
    finished = {(r['key'], r['checkpoint']) for r in doc['conditions']}
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, c, t, clips, args, output)
                   for c in conds for t in args.checkpoints if (c['key'], t) not in finished]
        for future in as_completed(futures):
            row = future.result()
            doc['conditions'].append(row)
            print(f"{row['key']:18} {row['checkpoint']:12} {row['carried']}/{len(clips)}  {row['wall_seconds']:.1f}s", flush=True)
            (output / 'partial.json').write_text(json.dumps(doc, indent=2))
    doc.update(complete=True, final_segment_wall_seconds=time.perf_counter()-start,
        wall_seconds=(datetime.now(timezone.utc)-datetime.fromisoformat(doc['provenance']['timestamp'])).total_seconds(),
        sweep_timing_definition='Elapsed wall time since first provenance capture, including interruption and resume gaps. Per-group wall_seconds excludes gaps.')
    (output / 'comparison.json').write_text(json.dumps(doc, indent=2))
    print(output / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
