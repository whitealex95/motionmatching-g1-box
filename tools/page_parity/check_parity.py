"""Compares mm_g1 with lmm-g1-box's page on the page's scripted sequence (kinematic, sbplace, Locomotion 'emmboth').

    node tools/page_parity/js_trace.mjs ~/Projects/lmm-g1-box /tmp/js.jsonl --states /tmp/states.jsonl
    python tools/page_parity/check_parity.py /tmp/js.jsonl                      # free run, frame by frame
    python tools/page_parity/check_parity.py /tmp/js.jsonl --forced /tmp/states.jsonl
    node tools/page_parity/js_match_cases.mjs ~/Projects/lmm-g1-box /tmp/cases.json
    python tools/page_parity/check_parity.py /tmp/cases.json --cases            # Controller.match alone

A free run follows the page bit for bit only until V8's sin, exp or atan2 rounds differently from the C library
(1 ulp), and then until a decision of the page that depends on such a bit: the page steers around the target box
itself, with a lateral offset of zero up to rounding. --forced loads the page's whole state before every frame,
steps once and compares with the page's state after it, so every frame is checked from the same state."""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from mm_g1.database import Pose  # noqa: E402
from mm_g1.mathq import qyaw  # noqa: E402
from mm_g1.sim import Sim  # noqa: E402


def frame_state(sim, k):
    c, b, P, w = sim.controller, sim.controller.backend, sim.controller.pose, sim.world
    return {'k': k, 'step': sim.scenario['index'], 'mode': c.mode, 'tag': c.boxTag, 'db': b.active, 'frame': int(b.frame),
            'row': int(b.db[b.active].idx[b.frame]), 'root': P.rootPos, 'yaw': qyaw(P.rootRot), 'hip': P.hipPos,
            'held': P.held, 'joints': sim.render_joints, 'holding': w.holding_id, 'pick': w.pick_id,
            'boxes': [[x['id'], x['pos'][0], x['pos'][1], x['yaw']] for x in w.boxes],
            'place': [*w.place_target['pos'], w.place_target['yaw']] if w.place_target else None,
            'target': c.target['side'] if c.target else None, 'cost': c.boxCost if math.isfinite(c.boxCost) else None}


def pose_from(d):
    p = Pose()
    for k, v in d.items():
        setattr(p, k, np.asarray(v, dtype=np.float64) if k in ('joints', 'jointVel') else v)
    return p


def num(v):
    return math.inf if v == 'inf' else v


def load_state(sim, S):
    c, b, w = sim.controller, sim.controller.backend, sim.world
    for k, v in S['c'].items():
        if k in ('curr', 'pose'):
            v = pose_from(v)
        elif k == 'off':
            v = {key: np.asarray(x, dtype=np.float64) if key in ('joints', 'jointVel') else x for key, x in v.items()}
        elif k == 'boxCost':
            v = num(v)
        setattr(c, k, copy.deepcopy(v))
    b.set_state((S['b']['active'], S['b']['frame'], S['b']['frac'], S['b']['loco']))
    W = S['w']
    w.boxes = copy.deepcopy(W['boxes'])
    w.next_id, w.pick_id, w.holding_id = W['nextId'], W['pickId'], W['holdingId']
    w.place_target, w.regrasp = copy.deepcopy(W['placeTarget']), W['regrasp']
    s = S['s']
    sim.scenario = copy.deepcopy(s['scenario'])
    sim.sim_time, sim.stats, sim.force_search = s['simTime'], dict(s['stats']), s['forceSearch']
    sim.grip, sim.parked, sim.held_pose = copy.deepcopy(s['grip']), copy.deepcopy(s['parked']), copy.deepcopy(s['heldPose'])
    sim.render_joints = None if s['renderJoints'] is None else np.asarray(s['renderJoints'])
    feet = copy.deepcopy(S['f']['feet'])
    for f in feet:
        f.setdefault('final', None)
    sim.foot_lock.feet, sim.foot_lock.skate = feet, dict(S['f']['skate'])


def flat(prefix, v, out):
    """Leaves of a state as {path: value}, numbers and strings alike."""
    if isinstance(v, Pose):
        v = {k: getattr(v, k) for k in Pose.__slots__}
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, dict):
        for k, x in v.items():
            flat(f'{prefix}.{k}', x, out)
    elif isinstance(v, (list, tuple)):
        for i, x in enumerate(v):
            flat(f'{prefix}[{i}]', x, out)
    else:
        out[prefix] = num(v) if isinstance(v, str) and v == 'inf' else v
    return out


def py_state(sim):
    c, b, w = sim.controller, sim.controller.backend, sim.world
    S = {'c': {k: getattr(c, k) for k in ('mode', 'boxTag', 'holding', 'sim', 'desiredRot', 'desiredVel', 'searchTimer',
                                         'forceSearch', 'holdSwitchTime', 'commit', 'target', 'boxCost', 'calledOff',
                                         'engaged', 'reapproach', 'leaveBox', 'missed', 'walkIn', 'navigating',
                                         'trajectory', 'curr', 'pose', 'off', 'tsrc', 'tdst', 'events', 'semantics')},
         'b': {'active': b.active, 'frame': int(b.frame), 'frac': b.frac, 'loco': b.loco},
         'w': {'boxes': w.boxes, 'nextId': w.next_id, 'pickId': w.pick_id, 'holdingId': w.holding_id,
               'placeTarget': w.place_target, 'regrasp': w.regrasp},
         's': {'scenario': sim.scenario, 'simTime': sim.sim_time, 'stats': sim.stats, 'forceSearch': sim.force_search,
               'grip': sim.grip, 'parked': sim.parked, 'heldPose': sim.held_pose, 'renderJoints': sim.render_joints},
         'f': {'feet': [{k: v for k, v in f.items() if not (k == 'final' and v is None)} for f in sim.foot_lock.feet],
               'skate': sim.foot_lock.skate}}
    return flat('', S, {})


def forced(states_path, tol):
    sim = Sim.build(loco='emmboth')
    sim.box_size = [sim.cfg['data_box_size'][1], sim.cfg['data_box_size'][1], sim.cfg['data_box_size'][2]]
    sim.start_scripted()
    with open(states_path) as f:
        states = [json.loads(l) for l in f]
    worst, discrete, n = {}, [], 0
    for k in range(len(states) - 1):
        load_state(sim, states[k])
        sim.step()
        got, want = py_state(sim), flat('', states[k + 1], {})
        for key in sorted(set(got) | set(want)):
            a, b = got.get(key, '<missing>'), want.get(key, '<missing>')
            if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                if a != b and not (a is None and b == '<missing>') and not (b is None and a == '<missing>'):
                    discrete.append((k, key, a, b))
                continue
            if math.isinf(a) or math.isinf(b):
                if a != b:
                    discrete.append((k, key, a, b))
                continue
            group = key.split('[')[0]
            worst[group] = max(worst.get(group, (0.0, -1)), (abs(a - b), k))
        n += 1
    big = {g: v for g, v in worst.items() if v[0] > tol}
    print(f'{n} frames stepped from the page\'s state')
    for g, (d, k) in sorted(worst.items(), key=lambda x: -x[1][0])[:12]:
        print(f'  {g:40s} max |py - js| {d:.2e} (frame {k})')
    frames = sorted({d[0] for d in discrete})
    print(f'{len(discrete)} discrete differences in {len(frames)} frames')
    for d in discrete[:20]:
        print('  ', d)
    ok = not big and not discrete
    print('FORCED PARITY OK' if ok else 'FORCED PARITY: differences above')
    return 0 if ok else 1


def cases(path, tol):
    J = json.load(open(path))
    sim = Sim.build(loco='emmboth')
    c, b = sim.controller, sim.controller.backend
    bad = [key for key, m in (('sceneBotPre', b.scenebot_pre('box')), ('beforeEvent', b.before_event_mask('box')))
           if not np.array_equal(m, J[key])]
    worst = 0.0
    for k, cs in enumerate(J['cases']):
        b.active, b.frame, b.frac, c.walkIn = 'box', cs['frame'], 0.0, cs['walkIn']
        m = c.match(cs['kind'], np.asarray(cs['raw'], dtype=np.float64))
        if m['frame'] != cs['js']['frame']:
            bad.append((k, cs['kind'], cs['walkIn'], cs['frame'], m['frame'], cs['js']['frame']))
        else:
            worst = max(worst, abs(m['cost'] - cs['js']['cost']))
    print(f"{len(J['cases'])} searches, max |py - js| cost {worst:.2e}")
    for x in bad[:20]:
        print('  ', x if isinstance(x, str) else 'case %d %s walkIn=%s playing %d: py frame %d, js frame %d' % x)
    ok = not bad and worst <= tol
    print('MATCH PARITY OK' if ok else 'MATCH PARITY: differences above')
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--tol', type=float, default=1e-3, help='position (m) and joint (rad) tolerance')
    ap.add_argument('--forced', metavar='STATES', help="js_trace.mjs --states output: step from the page's state each frame")
    ap.add_argument('--cases', action='store_true', help="the trace is js_match_cases.mjs's output: compare Controller.match")
    a = ap.parse_args()
    if a.cases:
        return cases(a.trace, 1e-9)
    if a.forced:
        return forced(a.forced, 1e-9)
    js = [json.loads(l) for l in open(a.trace)]
    t0 = time.time()
    sim = Sim.build(loco='emmboth')
    sim.box_size = [sim.cfg['data_box_size'][1], sim.cfg['data_box_size'][1], sim.cfg['data_box_size'][2]]
    print(f'loaded in {time.time() - t0:.1f} s')
    sim.start_scripted()
    t0 = time.time()
    worst = {'root': 0.0, 'hip': 0.0, 'joints': 0.0, 'yaw': 0.0}
    first_bad = None
    for k, ref in enumerate(js):
        sim.step()
        s = frame_state(sim, k)
        bad = [key for key in ('step', 'mode', 'tag', 'db', 'row', 'holding', 'pick', 'target') if s[key] != ref[key]]
        d = {'root': max(abs(x - y) for x, y in zip(s['root'], ref['root'])),
             'hip': max(abs(x - y) for x, y in zip(s['hip'], ref['hip'])),
             'joints': float(np.max(np.abs(np.asarray(s['joints']) - np.asarray(ref['joints'])))),
             'yaw': abs(math.remainder(s['yaw'] - ref['yaw'], 2 * math.pi))}
        for key, v in d.items():
            worst[key] = max(worst[key], v)
            if v > a.tol:
                bad.append(f'{key} {v:.2e}')
        if bad and first_bad is None:
            first_bad = k
            print(f'frame {k}: {bad}')
            for key in ('step', 'mode', 'tag', 'db', 'frame', 'row', 'holding', 'pick', 'target', 'cost', 'root', 'yaw', 'held'):
                print(f'  {key:8s} py {s[key]!s:40.40s} js {ref[key]!s:40.40s}')
            break
    n = k + 1
    print(f'{n} of {len(js)} frames in {time.time() - t0:.1f} s, done={sim.done}, worst {worst}')
    print('PARITY OK' if first_bad is None and n == len(js) else f'DIVERGES at frame {first_bad}')
    return 0 if first_bad is None else 1


if __name__ == '__main__':
    sys.exit(main())
