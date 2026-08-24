"""Plots for the single-pick box stress test (reads out/stress_single/results.json)."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'stress_single')

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e8e7e4'
SERIES = '#2a78d6'
STATUS = {'SUCCESS': ('#0ca30c', '✓'),   # good, check
          'DROPPED': ('#fab219', '▽'),   # lifted but not set down flat
          'NO GRIP': ('#ec835a', '○'),   # never lifted
          'FELL': ('#d03b3b', '✕'),      # robot fell
          'PARTIAL': ('#ec835a', '◇')}   # everything else short of success

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'text.color': INK,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2,
    'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.axisbelow': True,
})


def outcome(r):
    if r['fallen']:
        return 'FELL'
    if not r['lifted']:
        return 'NO GRIP'
    if r['success']:
        return 'SUCCESS'
    if not r['placed_flat']:
        return 'DROPPED'
    return 'PARTIAL'


def load():
    with open(os.path.join(OUT, 'results.json')) as f:
        rows = json.load(f)
    for r in rows:
        r['outcome'] = outcome(r)
    return rows


def bars(ax, xs, rows, xlabel, thresholds=None):
    heights = [r['max_box_z'] for r in rows]
    ax.bar(range(len(xs)), heights, width=0.55, color=SERIES, zorder=2)
    for i, r in enumerate(rows):
        col, glyph = STATUS[r['outcome']]
        ax.text(i, heights[i] + 0.045, glyph, ha='center', va='bottom',
                color=col, fontsize=13, fontweight='bold', zorder=4)
    if thresholds is not None:
        for i, th in enumerate(thresholds):
            ax.plot([i - 0.38, i + 0.38], [th, th], ls=(0, (3, 2)),
                    color=INK2, lw=1.1, zorder=3)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 1.45)
    ax.grid(axis='x', visible=False)


def legend(fig, statuses, with_threshold=True):
    handles = [plt.Line2D([], [], ls='none', marker='$' + STATUS[s][1] + '$',
                          ms=9, color=STATUS[s][0], label=s.title())
               for s in statuses]
    if with_threshold:
        handles.append(plt.Line2D([], [], ls=(0, (3, 2)), color=INK2, lw=1.1,
                                  label='Lift threshold (rest + 0.25 m)'))
    fig.legend(handles=handles, loc='outside upper center', frameon=False,
               ncol=len(handles), fontsize=9)


def group(rows, g):
    return [r for r in rows if r['group'] == g]


def main():
    rows = load()
    used = set()

    # scale sweep
    rs = group(rows, 'scale')
    used.update(r['outcome'] for r in rs)
    fig, ax = plt.subplots(figsize=(6.4, 3.4), layout='constrained')
    xs = [f"{r['box_size'][0] / 0.15:g}×" for r in rs]
    bars(ax, xs, rs, 'uniform box scale (1× = 0.30 × 0.20 × 0.30 m)',
         thresholds=[r['box_size'][2] + 0.25 for r in rs])
    ax.set_ylabel('max box height (m)')
    ax.set_title('Box size (0.1 kg)')
    legend(fig, [s for s in STATUS if s in {r['outcome'] for r in rs}])
    fig.savefig(os.path.join(OUT, 'fig_scale.png'), dpi=160)

    # mass sweep
    rm = group(rows, 'mass')
    used.update(r['outcome'] for r in rm)
    fig, ax = plt.subplots(figsize=(6.4, 3.4), layout='constrained')
    xs = [f"{r['box_mass']:g}" for r in rm]
    bars(ax, xs, rm, 'box mass (kg), default 0.30 × 0.20 × 0.30 m box',
         thresholds=[0.15 + 0.25] * len(rm))
    ax.set_ylabel('max box height (m)')
    ax.set_title('Box mass')
    legend(fig, [s for s in STATUS if s in {r['outcome'] for r in rm}])
    fig.savefig(os.path.join(OUT, 'fig_mass.png'), dpi=160)

    # dimension sweeps: grip width / height / depth
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharey=True,
                             layout='constrained')
    panels = [('width', 'grip width 2·hy (m)',
               lambda r: f"{2 * r['box_size'][1]:g}"),
              ('height', 'height 2·hz (m)',
               lambda r: f"{2 * r['box_size'][2]:g}"),
              ('depth', 'depth 2·hx (m)',
               lambda r: f"{2 * r['box_size'][0]:g}")]
    for ax, (g, xlabel, fmt) in zip(axes, panels):
        rg = group(rows, g)
        used.update(r['outcome'] for r in rg)
        th = ([r['box_size'][2] + 0.25 for r in rg] if g == 'height'
              else [0.15 + 0.25] * len(rg))
        bars(ax, [fmt(r) for r in rg], rg, xlabel, thresholds=th)
        ax.set_title({'width': 'Grip width (hands close on ±y)',
                      'height': 'Box height', 'depth': 'Box depth'}[g])
    axes[0].set_ylabel('max box height (m)')
    legend(fig, [s for s in STATUS if s in used])
    fig.savefig(os.path.join(OUT, 'fig_dims.png'), dpi=160)

    # scale x mass outcome matrix
    scales = [0.8, 1.0, 1.2]
    masses = [0.1, 0.5, 1.0, 2.0]
    fig, ax = plt.subplots(figsize=(6.2, 3.8), layout='constrained')
    ax.grid(visible=False)
    for i, s in enumerate(scales):
        for j, m in enumerate(masses):
            match = [r for r in rows
                     if abs(r['box_size'][0] - 0.15 * s) < 1e-6
                     and abs(r['box_size'][1] - 0.10 * s) < 1e-6
                     and r['box_mass'] == m]
            if not match:
                continue
            o = match[0]['outcome']
            col, glyph = STATUS[o]
            ax.add_patch(plt.Rectangle((j - 0.44, i - 0.44), 0.88, 0.88,
                                       facecolor=col, alpha=0.18,
                                       edgecolor=col, lw=1.2))
            ax.text(j, i, glyph, ha='center', va='center', color=col,
                    fontsize=15, fontweight='bold')
    ax.set_xticks(range(len(masses)))
    ax.set_xticklabels([f'{m:g} kg' for m in masses])
    ax.set_yticks(range(len(scales)))
    ax.set_yticklabels([f'{s:g}×' for s in scales])
    ax.set_xlim(-0.6, len(masses) - 0.4)
    ax.set_ylim(-0.6, len(scales) - 0.4)
    ax.set_xlabel('box mass')
    ax.set_ylabel('box scale')
    ax.set_title('Size × mass interaction')
    legend(fig, [s for s in STATUS if s in used], with_threshold=False)
    fig.savefig(os.path.join(OUT, 'fig_grid.png'), dpi=160)

    # markdown result table
    lines = ['| run | size (m, full) | mass | outcome | max box z | max xy err |',
             '|---|---|---|---|---|---|']
    for r in rows:
        if 'dedup_of' in r:
            continue
        sz = ' × '.join(f'{2 * v:g}' for v in r['box_size'])
        lines.append(f"| {r['name']} | {sz} | {r['box_mass']:g} kg | "
                     f"{r['outcome']} | {r['max_box_z']:.2f} | "
                     f"{r['max_xy_err']:.2f} |")
    with open(os.path.join(OUT, 'results_table.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('plots + table written to', OUT)


if __name__ == '__main__':
    main()
