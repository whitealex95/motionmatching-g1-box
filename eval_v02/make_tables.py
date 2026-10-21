#!/usr/bin/env python3
"""Write eval_v02/results/RESULTS.md and results/latex/squeeze_table.tex
from results/squeeze_results.json.

    python eval_v02/make_tables.py
"""
import argparse
import datetime
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
ORDER = ['none', 'shoulder', 'wrist', 'shoulder+wrist',
         'palm15', 'palm25', 'palm40', 'palm25+biases']
KINDS = ('success', 'slipped', 'dropped', 'fell')


def tally(row):
    t = {k: 0 for k in KINDS}
    for v in row['outcomes'].values():
        t[v] += 1
    return t


def tex_escape(s):
    return s.replace('_', r'\_')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=os.path.join(RESULTS,
                                                   'squeeze_results.json'))
    args = ap.parse_args()
    with open(args.json) as fh:
        doc = json.load(fh)

    rows = {(c['key'], c['checkpoint']): c for c in doc['conditions']}
    ckpts = doc['checkpoints']
    keys = [k for k in ORDER if any((k, c) in rows for c in ckpts)]
    keys += [c['key'] for c in doc['conditions']
             if c['key'] not in keys]
    n = max(c['n_clips'] for c in doc['conditions'])
    now = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    src = os.path.relpath(args.json, os.path.dirname(HERE))

    md = [f'# Squeeze test: clips carried out of {n}', '',
          f'Source: `{src}`  ', f'Sweep generated: {doc["generated"]}  ',
          f'Repo commit at sweep time: `{doc["git_commit"]}`  ',
          f'Tables generated: {now}', '',
          'Kinematic-reference tracking of the '
          f'{n} OmniRetarget G1 large-box carry clips with the SONIC tracker '
          'in MuJoCo physics and a frictional carton. Not the full '
          'pick-carry-place pipeline. `carried` counts clips meeting the '
          'success predicate; `slip` = never lifted, `drop` = lifted but not '
          'still held at the reference put-down, `fell` = robot pelvis below '
          '0.28 m.', '']
    head = ['Strategy'] + [f'{c} carried' for c in ckpts] + \
           [f'{c} slip/drop/fell' for c in ckpts]
    md.append('| ' + ' | '.join(head) + ' |')
    md.append('|' + '|'.join(['---'] * len(head)) + '|')
    for k in keys:
        label = next(rows[(k, c)]['label'] for c in ckpts if (k, c) in rows)
        cells = [f'{label} (`{k}`)']
        for c in ckpts:
            r = rows.get((k, c))
            cells.append(f'**{r["carried"]}**/{r["n_clips"]}' if r else '-')
        for c in ckpts:
            r = rows.get((k, c))
            if r is None:
                cells.append('-')
                continue
            t = tally(r)
            cells.append(f'{t["slipped"]}/{t["dropped"]}/{t["fell"]}')
        md.append('| ' + ' | '.join(cells) + ' |')
    md += ['', '## Per-clip outcomes', '']
    for c in ckpts:
        md += [f'### {c}', '',
               '| Clip | ' + ' | '.join(f'`{k}`' for k in keys) + ' |',
               '|' + '|'.join(['---'] * (len(keys) + 1)) + '|']
        for clip in doc['clips']:
            cells = [clip.replace('_original_mujoco', '')]
            for k in keys:
                r = rows.get((k, c))
                cells.append(r['outcomes'].get(clip, '-') if r else '-')
            md.append('| ' + ' | '.join(cells) + ' |')
        md.append('')
    md_path = os.path.join(RESULTS, 'RESULTS.md')
    with open(md_path, 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    print(f'wrote {md_path}')

    tex = [f'% source: {src}',
           f'% sweep generated: {doc["generated"]}  commit '
           f'{doc["git_commit"]}',
           f'% table generated: {now} by eval_v02/make_tables.py',
           r'\begin{tabular}{l' + 'cc' * len(ckpts) + '}',
           r'\toprule']
    top = ['Grasp strategy'] + [r'\multicolumn{2}{c}{\texttt{%s}}'
                                % tex_escape(c) for c in ckpts]
    tex += [' & '.join(top) + r' \\',
            r'\cmidrule(lr){2-3} \cmidrule(lr){4-5}',
            ' & '.join([''] + ['carried', 'slip/drop/fell'] * len(ckpts))
            + r' \\', r'\midrule']
    for k in keys:
        label = next(rows[(k, c)]['label'] for c in ckpts if (k, c) in rows)
        cells = [tex_escape(label)]
        for c in ckpts:
            r = rows.get((k, c))
            if r is None:
                cells += ['--', '--']
                continue
            t = tally(r)
            cells += [f'{r["carried"]}',
                      f'{t["slipped"]}/{t["dropped"]}/{t["fell"]}']
        tex.append(' & '.join(cells) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}']
    tex_dir = os.path.join(RESULTS, 'latex')
    os.makedirs(tex_dir, exist_ok=True)
    tex_path = os.path.join(tex_dir, 'squeeze_table.tex')
    with open(tex_path, 'w') as fh:
        fh.write('\n'.join(tex) + '\n')
    print(f'wrote {tex_path}')


if __name__ == '__main__':
    main()
