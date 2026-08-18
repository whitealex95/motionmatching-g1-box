"""Figures for the medicine-carton shape sweep (out/stress_shape)."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stress_plots import STATUS, bars, legend, outcome

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'stress_shape')


def load():
    with open(os.path.join(OUT, 'results.json')) as f:
        rows = json.load(f)
    for r in rows:
        r['outcome'] = outcome(r)
    return rows


def main():
    rows = load()
    used = set(r['outcome'] for r in rows)

    ru = [r for r in rows if r['group'] == 'uniform']
    fig, ax = plt.subplots(figsize=(6.8, 3.4), layout='constrained')
    xs = [f"{round(r['box_size'][1] / 0.1695, 2):g}×" for r in ru]
    bars(ax, xs, ru, 'uniform scale of the carton (1× = 0.32 × 0.34 × 0.36 m)',
         thresholds=[r['box_size'][2] + 0.25 for r in ru])
    ax.set_ylabel('max box height (m)')
    ax.set_title('Uniform scale (0.5 kg)')
    legend(fig, [s for s in STATUS if s in {r['outcome'] for r in ru}])
    fig.savefig(os.path.join(OUT, 'fig_uniform.png'), dpi=160)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4), sharey=True,
                             layout='constrained')
    panels = [('axis_x', 0, 'depth (m, toward the robot)'),
              ('axis_y', 1, 'grip width (m, palms close on ±y)'),
              ('axis_z', 2, 'height (m)')]
    for ax, (g, ai, xlabel) in zip(axes, panels):
        rg = sorted((r for r in rows if r['group'] == g),
                    key=lambda r: r['box_size'][ai])
        bars(ax, [f"{2 * r['box_size'][ai]:.2f}" for r in rg], rg, xlabel,
             thresholds=[r['box_size'][2] + 0.25 for r in rg])
        ax.set_title({'axis_x': 'Depth only', 'axis_y': 'Grip width only',
                      'axis_z': 'Height only'}[g])
    axes[0].set_ylabel('max box height (m)')
    legend(fig, [s for s in STATUS if s in used])
    fig.savefig(os.path.join(OUT, 'fig_axes.png'), dpi=160)
    print('figures written to', OUT)


if __name__ == '__main__':
    main()
