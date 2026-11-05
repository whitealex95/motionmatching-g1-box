#!/usr/bin/env python3
"""Render matched locomotion commands through both physical trackers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--loco-root', type=Path, default=ROOT.parent / 'motionmatching-g1-loco')
    ap.add_argument('--python', default=str(Path.home() / 'miniconda3/envs/mm-g1/bin/python'))
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--scene', default='00_maindemo')
    ap.add_argument('--out', type=Path, default=ROOT / 'eval_supermimic/results/loco')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    def run(tracker):
        video = args.out.resolve() / (tracker + '.mp4')
        metrics = args.out.resolve() / (tracker + '.json')
        if video.exists() or metrics.exists():
            raise RuntimeError(f'{video} already exists. Choose a new --out.')
        command = [args.python, '-u', '-m', 'sonic_tracking.interactive',
            '--scene', args.scene, '--controller', 'emm', '--tracker', tracker,
            '--auto', '--record', str(video), '--metrics-out', str(metrics),
            '--max-seconds', str(args.seconds), '--ref-mode', 'snap-xy',
            '--lookahead', '0.4', '--seed', '0', '--device', 'cpu',
            '--width', '1280', '--height', '720']
        start = time.perf_counter()
        proc = subprocess.run(command, cwd=args.loco_root,
            env=dict(os.environ, MUJOCO_GL='egl', OMP_NUM_THREADS='1', SONIC_ORT_THREADS='2',
                     SUPERMIMIC_TOOLS=str(ROOT / 'tools')),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        wall = time.perf_counter()-start
        (args.out / (tracker+'.log')).write_text(json.dumps(command)+'\n'+proc.stdout)
        if proc.returncode:
            raise RuntimeError(f'{tracker} failed: {proc.stdout[-2000:]}')
        result = json.loads(metrics.read_text())
        result.update(command=command, wall_seconds=wall,
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.loco_root,text=True).strip(),
            source_sha256={str(p.relative_to(args.loco_root)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (args.loco_root / 'sonic_tracking').glob('*.py')})
        metrics.write_text(json.dumps(result, indent=2))
        print(tracker, result, flush=True)
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ['sonic', 'supermimic']))
    (args.out / 'comparison.json').write_text(json.dumps(dict(
        protocol='Same EMM autopilot, scene, seed, reference correction and lookahead. Closed-loop references can diverge with physical feedback.',
        conditions=results), indent=2))


if __name__ == '__main__':
    main()
