#!/usr/bin/env python3
"""Create labelled, synchronized MP4 comparisons from measured rollouts."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def duration(path):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries',
        'format=duration','-of','default=noprint_wrappers=1:nokey=1',str(path)],text=True))


def stack(paths, titles, subtitle, output, width):
    lengths=[duration(p) for p in paths]
    longest=max(lengths)
    height=width*9//16
    cmd=['ffmpeg','-y','-v','error','-filter_complex_threads','2']
    for p in paths:cmd+=['-threads','2','-i',str(p)]
    filters=[]
    for i,title in enumerate(titles):
        filters.append(f'[{i}:v]fps=25,scale={width}:{height},setsar=1,'
            f'tpad=stop_mode=clone:stop_duration={longest-lengths[i]+.1},'
            f'trim=duration={longest},setpts=PTS-STARTPTS,'
            f'pad={width}:{height+48}:0:48:color=0x172528,'
            f"drawtext=fontfile={FONT}:text='{title}':fontcolor=white:fontsize=17:x=14:y=14[v{i}]")
    inputs=''.join(f'[v{i}]' for i in range(len(paths)))
    filters.append(f'{inputs}hstack=inputs={len(paths)},'
        f'pad=iw:ih+38:0:0:color=0x172528,'
        f"drawtext=fontfile={FONT}:text='{subtitle}':fontcolor=white:fontsize=15:x=16:y=h-27[out]")
    cmd+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264',
          '-threads','2','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    subprocess.run(cmd,check=True)
    return dict(command=cmd,source_durations_s=lengths,output=str(output))


def main():
    records=[]
    video=json.loads((RESULTS/'box_videos/comparison.json').read_text())
    clips=[]
    for key,label in [('none','No squeeze'),('shoulder+wrist','Shoulder 0.6 rad + wrist 0.2 rad'),('palm25','Palm force 25 N')]:
        rows=[next(r for r in video['conditions'] if r['key']==key and r['checkpoint']==t)
              for t in ['release','sonic_v1_1','supermimic']]
        paths=[r['episodes'][0]['video'] for r in rows]
        titles=[name+' | '+next(iter(r['outcomes'].values())) for name,r in zip(
            ['SONIC release','SONIC v1.1','SuperMimic demo'],rows)]
        output=RESULTS/('box_'+key.replace('+','_')+'_comparison.mp4')
        records.append(stack(paths,titles,label+' | sub12_largebox_071 | box offset 0 m',output,480))
        clips.append(output)
    # Equal output geometry allows lossless concatenation of the three sections.
    listing=RESULTS/'box_video_sections.txt'
    listing.write_text(''.join("file '"+p.name+"'\n" for p in clips))
    cmd=['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(listing),
         '-c','copy','-movflags','+faststart',str(RESULTS/'box_comparison.mp4')]
    subprocess.run(cmd,check=True)
    records.append(dict(command=cmd))
    records.append(stack([RESULTS/'loco/sonic.mp4',RESULTS/'loco/supermimic.mp4'],
        ['SONIC release','SuperMimic demo'],
        'EMM autopilot | seed 0 | snap-xy | 60 seconds | independent physical feedback',
        RESULTS/'loco_comparison.mp4',640))
    (RESULTS/'video_commands.json').write_text(json.dumps(records,indent=2))
    print(RESULTS/'box_comparison.mp4')
    print(RESULTS/'loco_comparison.mp4')


if __name__=='__main__':main()
