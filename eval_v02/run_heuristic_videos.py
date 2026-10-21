#!/usr/bin/env python3
"""Render one clip under three grasp heuristics and pull matched frames.

Three runs of sonic_g1_box/run_track_clips.py with --video on the SAME clip and
the `release` checkpoint: no heuristic, joint squeeze, virtual palm force.  The
MP4s go to results/videos/, and the same five reference timestamps are
extracted from each to results/frames/<setup>_<k>.png so make_plot.py can
assemble the strip.

    MUJOCO_GL=egl python eval_v02/run_heuristic_videos.py --clip <stem>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, 'results')
VIDEOS = os.path.join(RESULTS, 'videos')
FRAMES = os.path.join(RESULTS, 'frames')
TRACKER = os.path.join(ROOT, 'sonic_g1_box', 'run_track_clips.py')
OUT_DIR = os.path.join(ROOT, 'sonic_g1_box', 'out', 'track_clips')

CHECKPOINT = 'release'
SETTLE_TICKS = 50          # --settle 1.0 s at CONTROL_DT = 0.02 s
FPS = 50.0

SETUPS = [
    dict(key='none', label='no heuristic',
         args=['--shoulder-squeeze', '0', '--wrist-squeeze', '0',
               '--palm-force', '0']),
    dict(key='joint', label='joint squeeze',
         args=['--shoulder-squeeze', '0.6', '--wrist-squeeze', '0.2',
               '--palm-force', '0']),
    dict(key='palm', label='virtual palm force',
         args=['--shoulder-squeeze', '0', '--wrist-squeeze', '0',
               '--palm-force', '25']),
]
COMMON = ['--box-mass', '0.25']


def key_frames(stem):
    """The five reference 50 Hz frames the strip samples, from the clip data
    alone (no physics): settle end, first hand-contact label, lift, mid carry,
    reference put-down."""
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'sonic_g1_box'))
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import numpy as np
    from mm_g1 import config as C
    from mm_g1 import labels
    from mm_g1.data import _load_box_npz
    from run_track_clips import ClipMotion

    robot_q, box_pose = _load_box_npz(stem)
    _, _, contact = labels.box_labels(stem, box_pose, C.BOX_DATA_DIR)
    m = ClipMotion(robot_q, box_pose, speed=1.0)
    z_rest = float(np.median(box_pose[:C.BOX_REST_FRAMES, 2]))
    ref_z = m.qpos[:, 38]
    hi = np.flatnonzero(ref_z > z_rest + 0.25)
    lift = int(hi[0]) if len(hi) else 0
    carry_end = int(hi[-1]) if len(hi) else m.timesteps - 1
    on = (contact[m.src30, 2] > 0.5) | (contact[m.src30, 3] > 0.5)
    c = np.flatnonzero(on)
    first_contact = int(c[0]) if len(c) else lift
    return [('settle end', 0), ('contact', first_contact), ('lift', lift),
            ('mid carry', (lift + carry_end) // 2),
            ('put-down', carry_end)]


def run_one(setup, stem, python):
    for f in os.listdir(OUT_DIR) if os.path.isdir(OUT_DIR) else []:
        if f.startswith(stem) and f'_{CHECKPOINT}_' in f and f.endswith('.mp4'):
            os.remove(os.path.join(OUT_DIR, f))
    cmd = [python, TRACKER, '--sonic', CHECKPOINT, '--clips', stem,
           '--video'] + COMMON + setup['args']
    env = dict(os.environ, MUJOCO_GL='egl', CUDA_VISIBLE_DEVICES='')
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    log = os.path.join(RESULTS, 'logs', f'video_{setup["key"]}.log')
    with open(log, 'w') as fh:
        fh.write('# ' + ' '.join(cmd) + f'\n# exit {proc.returncode} '
                 f'wall {wall:.1f}s\n\n' + proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f'{setup["key"]} exited {proc.returncode}; see {log}')
    src = [f for f in os.listdir(OUT_DIR)
           if f.startswith(stem) and f'_{CHECKPOINT}_' in f
           and f.endswith('.mp4')]
    if len(src) != 1:
        raise RuntimeError(f'expected one mp4 for {setup["key"]}, got {src}')
    dst = os.path.join(VIDEOS, f'{stem}_{setup["key"]}.mp4')
    shutil.move(os.path.join(OUT_DIR, src[0]), dst)
    status = [l.strip() for l in proc.stdout.splitlines() if stem in l]
    return dst, wall, (status[0] if status else ''), ' '.join(cmd)


def extract(video, key, frames):
    import imageio.v2 as imageio
    want = {SETTLE_TICKS + k: i for i, (_, k) in enumerate(frames)}
    rd = imageio.get_reader(video)
    n = 0
    got = {}
    for idx, im in enumerate(rd):
        n = idx + 1
        if idx in want:
            got[want[idx]] = im
    rd.close()
    out = []
    for i, (name, k) in enumerate(frames):
        im = got.get(i)
        if im is None:                       # episode cut short (a fall)
            im = _last(video)
        path = os.path.join(FRAMES, f'{key}_{i}.png')
        imageio.imwrite(path, im)
        out.append(dict(index=i, name=name, ref_frame=k,
                        t=round(k / FPS, 2), path=os.path.relpath(path, ROOT),
                        available=(got.get(i) is not None)))
    return out, n


def _last(video):
    import imageio.v2 as imageio
    rd = imageio.get_reader(video)
    im = None
    for im in rd:
        pass
    rd.close()
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', required=True)
    ap.add_argument('--python', default=sys.executable)
    args = ap.parse_args()
    os.makedirs(VIDEOS, exist_ok=True)
    os.makedirs(FRAMES, exist_ok=True)
    os.makedirs(os.path.join(RESULTS, 'logs'), exist_ok=True)

    frames = key_frames(args.clip)
    print('reference frames:', frames)
    rec = dict(clip=args.clip, checkpoint=CHECKPOINT, fps=FPS,
               settle_ticks=SETTLE_TICKS,
               columns=[dict(name=n, ref_frame=k, t=round(k / FPS, 2))
                        for n, k in frames], setups=[])
    for s in SETUPS:
        video, wall, status, cmd = run_one(s, args.clip, args.python)
        shots, n = extract(video, s['key'], frames)
        size_mb = round(os.path.getsize(video) / 1e6, 1)
        print(f'  {s["key"]:6s} {wall:5.1f}s  {n} video frames  '
              f'{size_mb} MB  {status}')
        rec['setups'].append(dict(
            key=s['key'], label=s['label'], args=COMMON + s['args'],
            command=cmd, video=os.path.relpath(video, ROOT),
            video_mb=size_mb, video_frames=n, tracker_line=status,
            wall_seconds=round(wall, 1), frames=shots))
    out = os.path.join(RESULTS, 'heuristic_frames.json')
    with open(out, 'w') as fh:
        json.dump(rec, fh, indent=2)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
