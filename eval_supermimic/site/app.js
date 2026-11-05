const $ = id => document.getElementById(id);
const order=['release','sonic_v1_1','supermimic','supermimic_before_force'];
const names={release:'SONIC release',sonic_v1_1:'SONIC v1.1',supermimic:'SuperMimic · clip after force',supermimic_before_force:'SuperMimic · clip before force'};
const conditions=[['none','No squeeze'],['shoulder','Shoulder · 0.6 rad'],['wrist','Wrist · 0.2 rad'],['shoulder+wrist','Shoulder + wrist'],['palm15','Palm force · 15 N'],['palm25','Palm force · 25 N'],['palm40','Palm force · 40 N'],['palm60','Palm force · 60 N'],['palm25+biases','Palm 25 N + shoulder + wrist']];
let data;
function cell(row,text){const td=document.createElement('td');td.textContent=text;row.append(td);return td;}
function badge(td,status){td.textContent='';const s=document.createElement('span');s.className='badge '+status;s.textContent=status;td.append(s);}
function outcomes(){
  const key=$('outcome-condition').value, body=$('episodes').tBodies[0];body.replaceChildren();
  for(const clip of data.clips){const tr=document.createElement('tr');cell(tr,clip.replace('_original_mujoco',''));
    for(const tracker of order){const run=data.box.find(r=>r.key===key&&r.checkpoint===tracker);const td=cell(tr,'Pending');if(run){badge(td,run.outcomes[clip]);td.tabIndex=0;td.setAttribute('role','button');td.setAttribute('aria-label',`Watch ${clip}, ${names[tracker]}, ${key}`);td.onclick=()=>selectRun(key,tracker,clip);td.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();td.click();}};}}body.append(tr);}
}
function chooseVideos(){
  const key=$('heuristic').value, baseline=$('sonic-version').value, clip=$('clip').value;
  const supermimic=$('supermimic-version').value;
  $('supermimic-clip-label').textContent=supermimic==='supermimic'?'clip after force':'clip before force';
  for(const [id,tracker] of [['sonic',baseline],['supermimic',supermimic]]){
    const v=$('box-'+id);v.pause();const entry=data.box_videos.find(x=>x.key===key&&x.checkpoint===tracker&&x.clip===clip);
    if(entry){v.src=entry.video;$(`${id}-outcome`).textContent=entry.outcome;}
    else{v.removeAttribute('src');$(`${id}-outcome`).textContent='Pending';}v.load();
  }
  $('box-play').textContent='Play both';$('box-seek').value=0;$('box-time').value='0.00 s';
  $('outcome-condition').value=key;outcomes();
  for(const td of $('summary').querySelectorAll('td[data-key]'))td.classList.toggle('selected',td.dataset.key===key);
  $('box').querySelector('.note').setAttribute('data-clip',clip);
}
function selectRun(key,tracker,clip=$('clip').value){
  $('heuristic').value=key;$('clip').value=clip;
  if(tracker.startsWith('supermimic'))$('supermimic-version').value=tracker;
  else $('sonic-version').value=tracker;
  chooseVideos();$('box').scrollIntoView({behavior:'smooth'});
}
function transport(task){
  const videos=[$(task+'-sonic'),$(task+'-supermimic')], button=$(task+'-play'), seek=$(task+'-seek');
  const pause=()=>{videos.forEach(v=>v.pause());button.textContent='Play both';};
  button.onclick=async()=>{
    if(videos.some(v=>!v.paused)){pause();return;}
    if(!videos.every(v=>v.getAttribute('src')))return;
    const t=Number(seek.value);videos.forEach(v=>v.currentTime=Math.min(t,Math.max(0,v.duration-.04)));
    await Promise.all(videos.map(v=>v.play().catch(()=>{})));button.textContent='Pause both';
  };
  $(task+'-reset').onclick=()=>{pause();videos.forEach(v=>v.currentTime=0);seek.value=0;$(task+'-time').value='0.00 s';};
  seek.oninput=()=>{pause();videos.forEach(v=>{if(Number.isFinite(v.duration))v.currentTime=Math.min(Number(seek.value),Math.max(0,v.duration-.04));});$(task+'-time').value=Number(seek.value).toFixed(2)+' s';};
  for(const v of videos){v.onloadedmetadata=()=>{seek.max=Math.max(...videos.map(x=>Number.isFinite(x.duration)?x.duration:0));};}
  function frame(){
    const active=videos.find(v=>!v.paused&&!v.ended);
    if(active){seek.value=active.currentTime;$(task+'-time').value=active.currentTime.toFixed(2)+' s';
      const other=videos.find(v=>v!==active);
      if(!other.paused&&!other.ended&&Math.abs(other.currentTime-active.currentTime)>.15)other.currentTime=Math.min(active.currentTime,other.duration);
    }else if(videos.every(v=>v.paused||v.ended))button.textContent='Play both';
    requestAnimationFrame(frame);
  }frame();
}
async function init(){
  data=await fetch('data.json').then(r=>{if(!r.ok)throw Error('Results file unavailable');return r.json();});
  $('status').textContent=data.complete?`Completed · ${data.box.length*data.clips.length} box episodes · ${data.box_videos.length} videos · ${data.loco.length} locomotion runs`:'Evaluation in progress. Pending results are labelled below.';
  for(const clip of data.clips){const option=document.createElement('option');option.value=clip;option.textContent=clip.replace('_original_mujoco','');$('clip').append(option);}
  $('clip').value=data.clips.includes('sub12_largebox_071_original_mujoco')?'sub12_largebox_071_original_mujoco':data.clips[0];
  const head=$('summary').tHead.rows[0];
  for(const title of ['Controller / clipping',...conditions.map(x=>x[1])]){const th=document.createElement('th');th.textContent=title;head.append(th);}
  for(const [key,label] of conditions){
    for(const id of ['outcome-condition','heuristic']){const option=document.createElement('option');option.value=key;option.textContent=label;$(id).append(option);}
  }
  for(const tracker of order){
    const tr=document.createElement('tr');cell(tr,names[tracker]);
    for(const [key,label] of conditions){const row=data.box.find(r=>r.key===key&&r.checkpoint===tracker),td=cell(tr,'Pending');td.dataset.key=key;
      if(row){const button=document.createElement('button');button.className='result-cell';button.textContent=`${row.carried} / ${row.n_clips}`;button.setAttribute('aria-label',`${names[tracker]}, ${label}: ${row.carried} of ${row.n_clips}. Watch selected clip.`);button.onclick=()=>selectRun(key,tracker);td.replaceChildren(button);}}
    $('summary').tBodies[0].append(tr);
  }
  $('heuristic').value='palm25';
  $('outcome-condition').onchange=outcomes;
  for(const id of ['heuristic','sonic-version','supermimic-version','clip'])$(id).onchange=chooseVideos;
  chooseVideos();
  if(data.box_motionmatching){$('motion-matching').hidden=false;$('box-mm-video').src=data.box_motionmatching;}
  for(const tracker of ['sonic','supermimic']){const row=data.loco.find(x=>x.settings.tracker===tracker);if(row)$('loco-'+tracker).src=row.video;}
  const metrics=[['Recorded duration',x=>x.seconds.toFixed(1)+' s'],['Robot fell',x=>x.fallen?'Yes':'No'],['Waypoint index',x=>x.waypoint+' / 13'],['Tour completed',x=>x.completed?'Yes':'No'],['Robot path',x=>x.path_m.toFixed(2)+' m'],['Mean joint error',x=>x.joint_error_mean_rad.toFixed(4)+' rad'],['95th percentile joint error',x=>x.joint_error_p95_rad.toFixed(4)+' rad'],['Mean obstacle penetration',x=>x.penetration_mean_m.toFixed(3)+' m']];
  for(const [name,format] of metrics){const tr=document.createElement('tr');cell(tr,name);for(const tracker of ['sonic','supermimic']){const row=data.loco.find(x=>x.settings.tracker===tracker);cell(tr,row?format(row):'Pending');}$('loco-metrics').tBodies[0].append(tr);}
  for(const text of data.protocol){const p=document.createElement('p');p.textContent=text;$('box-protocol').append(p);}
  for(const link of data.downloads){const a=document.createElement('a');a.href=link.href;a.textContent=link.label+' ↗';a.download='';$('downloads').append(a);}
  document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{
    document.querySelectorAll('video').forEach(v=>v.pause());
    for(const name of ['box','loco'])$(name).hidden=name!==b.dataset.tab;
    document.querySelectorAll('[data-tab]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});
  });
  transport('box');transport('loco');
}
init().catch(e=>{$('status').textContent='Could not load results: '+e.message;});
