"""Box stress test for the motion-matched single-pick, dynamic (grasp) box.

Sweeps the free box's size and mass through run_mm_pickup.py's
B -> move-to-pick -> pick -> carry -> place sequence (SceneBot policy,
friction-only grasp, open loop) and collects the [stress-json] metrics.
One low-res video per run is kept in out/stress_single/, verdict in the
filename. The reference motion and reference box stay data-sized; only the
physical box changes.

Groups (base box 0.30 x 0.20 x 0.30 m, 0.1 kg -- the SceneBot demo's):
  scale   uniform scaling of the whole box
  mass    default box, heavier and heavier
  width   grip width (y half extent)
  height  box height (z half extent)
  depth   x half extent (toward the robot)
  grid    coarse scale x mass interaction
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, 'out', 'stress_single')
BASE = (0.15, 0.10, 0.15)


def cfgs():
    out = []
    for s in [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
        out.append(('scale', f'scale_{s:g}',
                    (BASE[0] * s, BASE[1] * s, BASE[2] * s), 0.1))
    for m in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]:
        out.append(('mass', f'mass_{m:g}kg', BASE, m))
    for y in [0.05, 0.075, 0.10, 0.15, 0.20]:
        out.append(('width', f'width_{2*y:g}m', (0.15, y, 0.15), 0.1))
    for z in [0.05, 0.10, 0.15, 0.25, 0.35]:
        out.append(('height', f'height_{2*z:g}m', (0.15, 0.10, z), 0.1))
    for x in [0.05, 0.10, 0.15, 0.25]:
        out.append(('depth', f'depth_{2*x:g}m', (x, 0.10, 0.15), 0.1))
    for s in [0.8, 1.0, 1.2]:
        for m in [0.5, 1.0, 2.0]:
            out.append(('grid', f'grid_s{s:g}_m{m:g}kg',
                        (BASE[0] * s, BASE[1] * s, BASE[2] * s), m))
    return out


def run_one(name, size, mass):
    video = os.path.join(OUT, name + '.mp4')
    cmd = [PY, os.path.join(HERE, 'run_mm_pickup.py'),
           '--mode', 'grasp', '--arm-gain', '1', '--squeeze', '0',
           '--box-size', *[f'{v:.4f}' for v in size],
           '--box-mass', f'{mass:g}', '--max-seconds', '50',
           '--video', video, '--width', '640', '--height', '360']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    for line in r.stdout.splitlines():
        if line.startswith('[stress-json] '):
            return json.loads(line[len('[stress-json] '):])
    raise RuntimeError(f'{name}: no metrics line\n{r.stdout[-2000:]}'
                       f'\n{r.stderr[-2000:]}')


def main():
    os.makedirs(OUT, exist_ok=True)
    results, cache = [], {}
    todo = cfgs()
    t0 = time.time()
    for i, (group, name, size, mass) in enumerate(todo):
        key = (tuple(round(v, 4) for v in size), mass)
        if key in cache:
            rec = dict(cache[key])
            rec.update(group=group, name=name, dedup_of=cache[key]['name'])
        else:
            rec = run_one(name, size, mass)
            rec.update(group=group, name=name)
            cache[key] = rec
            verdict = 'SUCCESS' if rec['success'] else 'FAIL'
            final = os.path.join(OUT, f'{name}_{verdict}.mp4')
            os.replace(os.path.join(OUT, name + '.mp4'), final)
            rec['video'] = os.path.basename(final)
        results.append(rec)
        print(f'[{i + 1:2d}/{len(todo)}] {name:16s} '
              f'{"SUCCESS" if rec["success"] else "fail":8s} '
              f'lifted={rec["lifted"]} flat={rec["placed_flat"]} '
              f'fallen={rec["fallen"]} max_z={rec["max_box_z"]:.2f} '
              f'({time.time() - t0:.0f}s)', flush=True)
    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1)
    n = sum(r['success'] for r in results)
    print(f'done: {n}/{len(results)} SUCCESS -> {OUT}/results.json')


if __name__ == '__main__':
    main()
