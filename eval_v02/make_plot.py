#!/usr/bin/env python3
"""Figures for the squeeze test.

`plot()`  -- grouped column chart of clips carried out of 20: x = grasp
            strategy, one column per SONIC checkpoint, count on each cap.
            Readable at 8 cm column width; light and dark versions from the
            same ramps.
`strip()` -- 3-row filmstrip of one clip under three grasp heuristics at
            matched reference timestamps, from the frames that
            run_heuristic_videos.py extracted.

    python eval_v02/make_plot.py            # both figures, if inputs exist
    python eval_v02/make_plot.py --only strip
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, 'results')

# Categorical slots 1-2 of the validated reference palette, light and dark
# steps; both modes pass every check of scripts/validate_palette.js.
THEME = {
    'light': dict(surface='#fcfcfb', primary='#0b0b0b', secondary='#52514e',
                  muted='#898781', grid='#e1e0d9', axis='#c3c2b7',
                  series=('#2a78d6', '#eb6834')),
    'dark': dict(surface='#1a1a19', primary='#f0efec', secondary='#c3c2b7',
                 muted='#898781', grid='#2c2c2a', axis='#383835',
                 series=('#3987e5', '#d95926')),
}

# Short two-line x labels; the README registry carries the full flags.
SHORT = {
    'none': 'none',
    'shoulder': 'shoulder\n0.6',
    'wrist': 'wrist\n0.2',
    'shoulder+wrist': 'shoulder\n+ wrist',
    'palm15': 'palm\n15 N',
    'palm25': 'palm\n25 N',
    'palm40': 'palm\n40 N',
    'palm25+biases': 'palm 25\n+ biases',
}
ORDER = list(SHORT)


def rounded_column(ax, x, w, h, color, radius_px=4.0):
    """Column with a 4 px rounded cap and square corners at the baseline."""
    fig = ax.figure
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(0, 0), (radius_px, radius_px)])
    rx, ry = min(abs(x1 - x0), w / 2), abs(y1 - y0)
    ry = min(ry, h) if h > 0 else 0.0
    l, r, t = x - w / 2, x + w / 2, h
    if h <= 0:
        return
    verts = [(l, 0), (l, t - ry), (l, t), (l + rx, t),
             (r - rx, t), (r, t), (r, t - ry), (r, 0), (l, 0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
             Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor='none',
                           linewidth=0, zorder=3, clip_on=False))
    del fig


def plot(doc, mode, out_stem):
    t = THEME[mode]
    rows = {(c['key'], c['checkpoint']): c for c in doc['conditions']}
    ckpts = doc['checkpoints']
    keys = [k for k in ORDER if any((k, c) in rows for c in ckpts)]
    n_clips = max(c['n_clips'] for c in doc['conditions'])

    fig, ax = plt.subplots(figsize=(3.15, 2.05))
    fig.patch.set_facecolor(t['surface'])
    ax.set_facecolor(t['surface'])

    group, gap = 0.80, 0.045
    w = (group - gap * (len(ckpts) - 1)) / len(ckpts)
    for j, ck in enumerate(ckpts):
        off = -group / 2 + w / 2 + j * (w + gap)
        for i, key in enumerate(keys):
            row = rows.get((key, ck))
            if row is None:
                continue
            rounded_column(ax, i + off, w, row['carried'], t['series'][j])
            ax.text(i + off, row['carried'] + 0.45, str(row['carried']),
                    ha='center', va='bottom', fontsize=5.6,
                    color=t['secondary'])
        ax.plot([], [], marker='s', markersize=4.2, linestyle='none',
                color=t['series'][j], label=ck)

    ax.set_ylim(0, n_clips + 2.0)
    ax.set_xlim(-0.65, len(keys) - 0.35)
    ax.set_yticks(range(0, n_clips + 1, 5))
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([SHORT[k] for k in keys], fontsize=5.6,
                       color=t['secondary'], linespacing=1.15)
    ax.set_ylabel(f'clips carried (of {n_clips})', fontsize=6.4,
                  color=t['secondary'])
    ax.tick_params(axis='y', labelsize=5.8, colors=t['muted'], length=0,
                   pad=2)
    ax.tick_params(axis='x', length=0, pad=2.5)
    for lb in ax.get_yticklabels():
        lb.set_color(t['muted'])
    ax.yaxis.grid(True, color=t['grid'], linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(t['axis'])
    ax.spines['bottom'].set_linewidth(0.5)

    leg = ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.06), ncol=2,
                    frameon=False, fontsize=6.0, handletextpad=0.25,
                    columnspacing=0.9, borderpad=0.0, handlelength=0.9)
    for txt in leg.get_texts():
        txt.set_color(t['secondary'])

    fig.tight_layout(pad=0.35)
    for ext in ('png', 'pdf'):
        path = f'{out_stem}.{ext}'
        fig.savefig(path, dpi=400, facecolor=t['surface'])
        print(f'wrote {path}')
    plt.close(fig)


def strip(rec, mode, out_stem, crop=0.62):
    """3 rows (heuristic) x N columns (matched reference timestamps).

    `crop` keeps that fraction of the frame width, centred, so the robot and
    the box fill the tile; the renderer's amber reference skeleton and its
    reference box stay visible, which is what makes a slip readable.
    """
    import matplotlib.image as mpimg
    t = THEME[mode]
    cols = rec['columns']
    rows = rec['setups']
    tiles = [[mpimg.imread(os.path.join(ROOT, f['path'])) for f in r['frames']]
             for r in rows]
    h, w = tiles[0][0].shape[:2]
    x0 = int(w * (1 - crop) / 2)
    cw = int(w * crop)
    tiles = [[im[:, x0:x0 + cw] for im in row] for row in tiles]

    tile_in = 1.18
    fig_w = 0.62 + tile_in * len(cols)
    fig_h = 0.30 + tile_in * h / cw * len(rows)
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(t['surface'])
    gs = fig.add_gridspec(len(rows), len(cols), left=0.62 / fig_w, right=0.999,
                          top=0.999, bottom=0.30 / fig_h,
                          wspace=0.014, hspace=0.014)
    for i, row in enumerate(rows):
        for j in range(len(cols)):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(tiles[i][j])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if j == 0:
                ax.set_ylabel(row['label'], fontsize=6.4, labelpad=3,
                              color=t['secondary'])
            if i == len(rows) - 1:
                ax.set_xlabel(f'{cols[j]["name"]}\n{cols[j]["t"]:.2f} s',
                              fontsize=6.0, labelpad=2, color=t['muted'],
                              linespacing=1.2)
    for ext in ('png', 'pdf'):
        path = f'{out_stem}.{ext}'
        fig.savefig(path, dpi=300, facecolor=t['surface'])
        print(f'wrote {path}')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=os.path.join(RESULTS,
                                                   'squeeze_results.json'))
    ap.add_argument('--frames-json',
                    default=os.path.join(RESULTS, 'heuristic_frames.json'))
    ap.add_argument('--only', choices=('bar', 'strip'), default=None)
    ap.add_argument('--crop', type=float, default=0.62)
    args = ap.parse_args()
    if args.only != 'strip' and os.path.exists(args.json):
        with open(args.json) as fh:
            doc = json.load(fh)
        plot(doc, 'light', os.path.join(RESULTS, 'squeeze_bar'))
        plot(doc, 'dark', os.path.join(RESULTS, 'squeeze_bar_dark'))
    if args.only != 'bar' and os.path.exists(args.frames_json):
        with open(args.frames_json) as fh:
            rec = json.load(fh)
        strip(rec, 'light', os.path.join(RESULTS, 'fig_box_heuristics'),
              crop=args.crop)


if __name__ == '__main__':
    main()
