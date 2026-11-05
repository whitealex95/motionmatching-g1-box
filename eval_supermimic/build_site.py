#!/usr/bin/env python3
"""Build a self-contained local page from measured results and real videos."""
import argparse
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--partial', action='store_true')
    ap.add_argument('--box-dir', default='box_all_videos')
    args = ap.parse_args()
    site = HERE / 'site'
    media = site / 'media'
    media.mkdir(exist_ok=True)
    def read(folder):
        path=HERE / 'results' / folder / 'comparison.json'
        if not path.exists() and args.partial:
            path=path.with_name('partial.json')
        if path.exists():
            return json.loads(path.read_text())
        if args.partial:
            return dict(conditions=[], clips=[], definitions={}, complete=False)
        raise FileNotFoundError(path)
    box = videos = read(args.box_dir)
    loco = read('loco')
    if not args.partial:
        assert box['complete'] and len(box['conditions'])==36
        assert len(box['clips'])==20 and all(r['n_clips']==20 for r in box['conditions'])
        assert len(loco['conditions'])==2
        assert set(box['checkpoints']) == {'release','sonic_v1_1','supermimic','supermimic_before_force'}
        assert len({(r['key'],r['checkpoint']) for r in box['conditions']}) == 36
    def copy_video(source, name):
        source=Path(source)
        dest=media / name
        if not dest.exists() or source.stat().st_mtime_ns > dest.stat().st_mtime_ns:
            shutil.copy2(source,dest)
        return 'media/'+name
    available=[]
    for row in videos['conditions']:
        assert len(row['episodes']) == len(box['clips'])
        for episode in row['episodes']:
            name='box_'+row['key'].replace('+','_')+'_'+row['checkpoint']+'_'+episode['clip']+'.mp4'
            available.append(dict(key=row['key'],checkpoint=row['checkpoint'],clip=episode['clip'],
                outcome=row['outcomes'][episode['clip']],video=copy_video(episode['video'],name),
                torque=episode.get('torque')))
    if not args.partial:
        assert len(available) == 720
    locomotion=[]
    for row in loco['conditions']:
        row=dict(row)
        row['video']=copy_video(row['settings']['record'],'loco_'+row['settings']['tracker']+'.mp4')
        locomotion.append(row)
    downloads=[]
    for folder,label in [(args.box_dir,'All 720 recorded box episodes and commands'),('box','Original 480-episode evaluation'),('loco','Locomotion metrics and commands')]:
        source=HERE / 'results' / folder / 'comparison.json'
        if source.exists():
            name=folder+'_results.json';shutil.copy2(source,site/name)
            downloads.append(dict(href=name,label=label))
    for name,label in [('RESULTS.md','Full recording report'),('summary.csv','Carry counts CSV'),('video_manifest.json','Video hashes and metadata'),('validation.json','Recording validation')]:
        source=HERE/'results'/args.box_dir/name
        if source.exists():
            dest='all_videos_'+name;shutil.copy2(source,site/dest)
            downloads.append(dict(href=dest,label=label))
    sample_labels = {'box_comparison.mp4':'Sample clip montage: three heuristics',
        'box_none_comparison.mp4':'Sample clip: no squeeze',
        'box_palm25_comparison.mp4':'Sample clip: 25 N palm force',
        'box_shoulder_wrist_comparison.mp4':'Sample clip: shoulder and wrist squeeze'}
    for p in sorted((HERE/'results').glob('*_comparison.mp4')):
        downloads.append(dict(href=copy_video(p,p.name),
            label=sample_labels.get(p.name,p.stem.replace('_',' ').capitalize()+' video')))
    offset=box['definitions'].get('box_offset_m',0)
    doc=dict(complete=bool(box.get('complete')) and len(locomotion)==2,
        clips=box['clips'],box=box['conditions'],box_videos=available,loco=locomotion,
        conditions=box['definitions'].get('conditions', []),
        downloads=downloads,protocol=[
        f'Carton mass 0.25 kg, friction 1.5, scale 1.0. Physical box offset {offset:g} m along the initial robot heading. The reference remains unchanged.',
        'Reference: the 20 OmniRetarget G1 large-box clips, resampled from 30 to 50 Hz. One second of settling, playback speed 1.0, 20 physics substeps per 0.02 s control tick.',
        'Success: peak physical box height exceeds resting height + 0.25 m, pelvis never falls below 0.28 m, and box height at the reference carry end exceeds resting height + 0.15 m. Fell, dropped, and slipped are retained as separate outcomes.',
        'Both methods use the same SceneBot robot and contact heuristics. Each tracker uses its native PD gains. The SuperMimic rows compare clipping after virtual-force addition against clipping PD first and adding the virtual-force torque with no final clip. The before-force variant disables the final scene motor and joint actuator-force clips. SONIC retains its existing scene motor and joint actuator-force limits.',
        'The downloaded SuperMimic demo does not specify its paper training checkpoint identity. These results measure the demo actor transferred to this physical scene.',
        'Palm force is per labelled palm, directed toward the box centre, ramped over 0.15 s and mapped to arm torques by the palm Jacobian. 60 N is the requested virtual-force magnitude, not a measured contact force.',
        'All 20 clips are recorded for every displayed combination at 1280 × 720 and 25 fps. Policy control remains 50 Hz. Torque diagnostics measure MuJoCo applied joint actuator torque at every physics substep, including settling.',
        'All runs are inference evaluations. Recorded wall time includes startup, inference, physics, output, and optional rendering. Concurrent processes share CPU resources.'])
    full=HERE/'results/box_motionmatching_comparison.mp4'
    if full.exists():
        doc['box_motionmatching']=copy_video(full,full.name)
        source=HERE/'results/box_motionmatching.json'
        if source.exists():
            shutil.copy2(source,site/source.name)
            doc['downloads'].append(dict(href=source.name,label='Full box motion-matching commands and outcomes'))
    (site/'data.json').write_text(json.dumps(doc,indent=2)+'\n')
    print(site/'index.html')


if __name__=='__main__':main()
