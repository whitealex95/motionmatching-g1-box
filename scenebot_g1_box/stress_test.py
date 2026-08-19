"""Box stress test for the SceneBot pickup (their motion, floor scene).

Sweeps the free box's size, shape, and mass through run_pickup.py's
move -> pick -> carry -> drop script and collects the [stress-json]
metrics. One low-res video per run is kept in out/stress/.

Groups:
  scale   uniform scaling of the default 0.30 x 0.20 x 0.30 m box
  mass    default box, heavier and heavier
  width   grip width (y half extent; the hands close on the +-y faces)
  height  box height (z half extent; the hands grab ~0.19 m up)
  depth   x half extent (toward the robot)
  grid    coarse scale x mass interaction
  shape   cylinder and sphere instead of a box
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, 'out', 'stress')
BASE = (0.15, 0.10, 0.15)


def cfgs():
    out = []
    for s in [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
        out.append(('scale', f'scale_{s:g}', 'box',
                    (BASE[0] * s, BASE[1] * s, BASE[2] * s), 0.1))
    for m in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        out.append(('mass', f'mass_{m:g}kg', 'box', BASE, m))
    for y in [0.05, 0.075, 0.10, 0.15, 0.20, 0.25]:
        out.append(('width', f'width_{2*y:g}m', 'box', (0.15, y, 0.15), 0.1))
    for z in [0.05, 0.10, 0.15, 0.25, 0.35, 0.45]:
        out.append(('height', f'height_{2*z:g}m', 'box', (0.15, 0.10, z), 0.1))
    for x in [0.05, 0.10, 0.15, 0.25, 0.35]:
        out.append(('depth', f'depth_{2*x:g}m', 'box', (x, 0.10, 0.15), 0.1))
    for s in [0.8, 1.0, 1.2]:
        for m in [0.5, 2.0, 5.0]:
            out.append(('grid', f'grid_s{s:g}_m{m:g}kg', 'box',
                        (BASE[0] * s, BASE[1] * s, BASE[2] * s), m))
    out += [('shape', 'cyl_r0.10', 'cylinder', (0.10, 0.0, 0.15), 0.1),
            ('shape', 'cyl_r0.15', 'cylinder', (0.15, 0.0, 0.15), 0.1),
            ('shape', 'sph_r0.10', 'sphere', (0.10, 0.0, 0.0), 0.1),
            ('shape', 'sph_r0.15', 'sphere', (0.15, 0.0, 0.0), 0.1)]
    return out


def run_one(name, btype, size, mass):
    video = os.path.join(OUT, name + '.mp4')
    cmd = [PY, os.path.join(HERE, 'run_pickup.py'),
           '--box-type', btype,
           '--box-size', *[f'{v:.4f}' for v in size],
           '--box-mass', f'{mass:g}',
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
    for i, (group, name, btype, size, mass) in enumerate(todo):
        key = (btype, tuple(round(v, 4) for v in size), mass)
        if key in cache:
            rec = dict(cache[key])
            rec.update(group=group, name=name, dedup_of=cache[key]['name'])
        else:
            rec = run_one(name, btype, size, mass)
            rec.update(group=group, name=name)
            cache[key] = rec
            verdict = 'SUCCESS' if rec['success'] else 'FAIL'
            final = os.path.join(OUT, f'{name}_{verdict}.mp4')
            os.replace(os.path.join(OUT, name + '.mp4'), final)
            rec['video'] = os.path.basename(final)
        results.append(rec)
        print(f'[{i + 1:2d}/{len(todo)}] {name:16s} '
              f'{"SUCCESS" if rec["success"] else "fail":8s} '
              f'lifted={rec["lifted"]} dropped={rec["dropped"]} '
              f'fallen={rec["fallen"]} max_z={rec["max_box_z"]:.2f} '
              f'({time.time() - t0:.0f}s)', flush=True)
    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1)
    n = sum(r['success'] for r in results)
    print(f'done: {n}/{len(results)} SUCCESS -> {OUT}/results.json')


if __name__ == '__main__':
    main()
