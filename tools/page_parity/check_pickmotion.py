"""Runs mm_g1/pickmotion.py on the cases js_pickmotion.mjs dumped and compares the solved rows of the Animation
Database field by field against the page's.

Run: node tools/page_parity/js_pickmotion.mjs <lmm-g1-box> /tmp/jsout
     python tools/page_parity/check_pickmotion.py /tmp/jsout
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mujoco  # noqa: E402

from mm_g1.database import DATA, MMBackend  # noqa: E402
from mm_g1.pickmotion import MANUAL, PickMotion  # noqa: E402

GROUPS = [
    ('pelvis pos', ['hip_pos']),
    ('pelvis quat', ['hip_quat']),
    ('joints', ['joints']),
    ('box pos', ['box_pos']),
    ('box quat', ['box_quat']),
    ('pelvis vel', ['hip_vel']),
    ('pelvis ang vel', ['hip_ang']),
    ('joint vel', ['joint_vel']),
    ('root (untouched)', ['root_delta', 'root_vel', 'root_ang']),
    ('contacts, held', ['contacts', 'held']),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir', help='the directory js_pickmotion.mjs wrote')
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.dir, 'info.json')))
    clips = meta['clips']
    W = meta['width']
    sizes = [(c['stop'] - c['start']) * W for c in clips]

    def read(name):
        v = np.fromfile(os.path.join(a.dir, name), dtype='<f4')
        out, o = [], 0
        for n in sizes:
            out.append(v[o:o + n].reshape(-1, W))
            o += n
        return out

    backend = MMBackend()
    pm = PickMotion(backend.cfg, backend, os.path.join(DATA, 'physics', 'scene.xml'),
                    os.path.join(DATA, 'physics', 'sonic.json'))
    py_clips = [{k: c[k] for k in ('kind', 'row', 'mirror', 'start', 'stop', 'attach', 'release')} for c in pm.clips]
    print('MuJoCo: page (WASM) %s, python %s' % (meta['mujoco'], mujoco.__version__))
    print('node %s, variant %s' % (meta['node'], meta['variant']))
    print('clips match: %s' % (py_clips == clips))
    if py_clips != clips:
        print('  js %s\n  py %s' % (clips, py_clips))

    js_data = read('data.bin')
    for i, c in enumerate(clips):
        same = np.array_equal(js_data[i], pm.base[c['row']])
        print('data rows of clip %d (%s): byte-identical %s' % (c['row'], c['kind'], same))

    L = backend.cfg['pose_layout']
    worst = {}
    for case in meta['cases']:
        js = read(case['bin'])
        pm.restore()
        pm.settings['autoHeight'] = case['autoHeight']
        pm.settings['autoSize'] = case['autoSize']
        pm.manual = dict(MANUAL, grasp=[list(g) for g in MANUAL['grasp']])
        pm.manual.update(case['manual'] or {})
        pm.clear_cache()
        pm.solveMs = 0.0
        t0 = time.perf_counter()
        changed = pm.sync({'pick': case['support'], 'place': case['support']}, case['box'])
        wall = (time.perf_counter() - t0) * 1000.0
        py = [np.array(backend.pose[c['start']:c['stop']]) for c in clips]

        print('\n=== %s  support %s  box %s  autoSize %s  manual %s ===' % (case['name'], case['support'], case['box'], case['autoSize'], case['manual']))
        print('changed: js %s, py %s' % (case['changed'], changed))
        print('solve: js %.0f ms, py %.0f ms (wall %.0f ms)' % (case['solveMs'], pm.solveMs, wall))
        if case['name'] == 'floor':
            print('no-op: js rows == data %s, py rows == data %s' %
                  (all(np.array_equal(js[i], js_data[i]) for i in range(len(clips))),
                   all(np.array_equal(py[i], js_data[i]) for i in range(len(clips)))))
        print('%-20s %12s %12s %9s %9s' % ('field', 'max abs diff', 'max |value|', 'max ulp', 'n differ'))
        for label, keys in GROUPS:
            d = m = u = 0.0
            n = 0
            for i in range(len(clips)):
                for k in keys:
                    s, e = L[k]
                    x, y = js[i][:, s:e].astype(np.float64), py[i][:, s:e].astype(np.float64)
                    gap = np.abs(x - y)
                    d = max(d, float(np.max(gap)))
                    m = max(m, float(np.max(np.abs(x))))
                    u = max(u, float(np.max(gap / np.spacing(np.maximum(np.abs(js[i][:, s:e]), np.float32(1e-30))))))
                    n += int(np.count_nonzero(gap))
            worst[label] = max(worst.get(label, 0.0), d)
            print('%-20s %12.3e %12.3e %9.2f %9d' % (label, d, m, u, n))

        for kind in ('pick', 'place'):
            ji, pi = case['info'][kind], pm.info[kind]
            if ji is None or pi is None:
                print('%s: js %s, py %s' % (kind, ji, pi))
                continue
            print('%s: pelvisMin js %.6f py %.6f (d %.2e), basePelvisMin js %.6f py %.6f' %
                  (kind, ji['pelvisMin'], pi['pelvisMin'], abs(ji['pelvisMin'] - pi['pelvisMin']),
                   ji['basePelvisMin'], pi['basePelvisMin']))
            if ji['palmMiss'] and pi['palmMiss']:
                print('      palmMiss js %s py %s (d %.2e)' %
                      (['%.6f' % v for v in ji['palmMiss']], ['%.6f' % v for v in pi['palmMiss']],
                       max(abs(x - y) for x, y in zip(ji['palmMiss'], pi['palmMiss']))))
            print('      offset js %s py %s (d %.2e)' %
                  (['%.6f' % v for v in ji['offset']], ['%.6f' % v for v in pi['offset']],
                   max(abs(x - y) for x, y in zip(ji['offset'], pi['offset']))))
            if ji['solved'] and pi['solved']:
                js_s, py_s = ji['solved'], pi['solved']
                print('      frames js %d py %d, grip js %.6f py %.6f, feetLeft js %.6f py %.6f' %
                      (js_s['frames'], len(py_s['qpos']), js_s['grip'], py_s['grip'], js_s['feetLeft'], py_s['feetLeft']))
                if js_s['stand'] and py_s['stand']:
                    print('      stand js %s py %s' % (js_s['stand'], py_s['stand']))
        d = max(abs(x - y) for k in ('pick', 'place') for x, y in zip(case['offsets'][k], pm.offsets()[k]))
        print('offsets(): max abs diff %.3e' % d)
        dl = max(abs(case['lift'][str(g)] - pm.lift(g)) for g in meta['liftRows'])
        print('lift() on %d rows: max abs diff %.3e' % (len(meta['liftRows']), dl))
        print('lift() js %s' % ['%.4f' % case['lift'][str(g)] for g in meta['liftRows']])
        print('lift() py %s' % ['%.4f' % pm.lift(g) for g in meta['liftRows']])

    print('\nworst over every case:')
    for label, _ in GROUPS:
        print('  %-20s %12.3e' % (label, worst.get(label, 0.0)))
    pm.restore()


if __name__ == '__main__':
    main()
