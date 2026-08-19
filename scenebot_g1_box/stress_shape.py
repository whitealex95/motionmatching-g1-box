"""Shape-only stress test around the MEDICINE carton, at a fixed 0.5 kg.

Baseline = the carton's true oriented dimensions (0.324 x 0.339 x 0.363 m,
tools/make_medicine_box.py), swept two ways through run_pickup.py's
move -> pick -> carry -> drop script:
  uni_S   uniform scale S in 0.5 .. 1.4
  x_F     one axis scaled by F in 0.3 .. 2.0, other two at baseline
          (x = depth toward the robot, y = grip width the palms close on,
           z = height)

Every run keeps its video (with the box dimensions captioned on each
frame and a SUCCESS/FAIL end card) and its compiled scene MJCF, so the
box geometry of record is in out/stress_shape/<name>.xml.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, 'out', 'stress_shape')
BASE = (0.162, 0.1695, 0.1815)   # medicine carton OBB half extents
MASS = 0.5


def cfgs():
    out = []
    for s in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
        out.append(('uniform', f'uni_{s:g}',
                    tuple(v * s for v in BASE)))
    for ai, axis in enumerate('xyz'):
        for f in [0.3, 0.5, 0.75, 1.25, 1.5, 2.0]:
            size = list(BASE)
            size[ai] *= f
            out.append((f'axis_{axis}', f'{axis}_{f:g}', tuple(size)))
    return out


def run_one(name, size):
    video = os.path.join(OUT, name + '.mp4')
    cmd = [PY, os.path.join(HERE, 'run_pickup.py'),
           '--box-type', 'box',
           '--box-size', *[f'{v:.4f}' for v in size],
           '--box-mass', f'{MASS:g}',
           '--save-mjcf', os.path.join(OUT, name + '.xml'),
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
    for i, (group, name, size) in enumerate(todo):
        key = tuple(round(v, 4) for v in size)
        if key in cache:
            rec = dict(cache[key])
            rec.update(group=group, name=name, dedup_of=cache[key]['name'])
        else:
            rec = run_one(name, size)
            rec.update(group=group, name=name)
            cache[key] = rec
            # verdict goes in the filename
            verdict = 'SUCCESS' if rec['success'] else 'FAIL'
            final = os.path.join(OUT, f'{name}_{verdict}.mp4')
            os.replace(os.path.join(OUT, name + '.mp4'), final)
            rec['video'] = os.path.basename(final)
        results.append(rec)
        dims = ' x '.join(f'{2 * v:.2f}' for v in size)
        print(f'[{i + 1:2d}/{len(todo)}] {name:9s} ({dims} m) '
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
