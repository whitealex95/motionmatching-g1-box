// The box bridge's database page (web/index.html): lmm-g1-box's motions.js (v3.0) with one data variant and, above it,
// the bridge's controller now (GET /api/live of sonic_g1_box/db_page.py): the controller states, the clip the robot
// plays and the last matches.

import * as THREE from 'three';
import { Scene } from './scene.js';
import { BOX } from './sim.js';
import { MMBackend, loadMMData, loadLoco } from './controller.js';
import { q, v3 } from './math.js';
import { VARIANTS, VARIANT, DEFAULT_VARIANT } from './variants.js';
import { StatePanel, KINDS, stateOf } from './states.js';

const FPS = 60;
const EMM_CARRY_DIR = 'data/emm_carry';
const DBS = ['loco', 'pick', 'place', 'clips'];
const KIND_LABEL = { ...KINDS, clip: 'Whole clip' };
const LIVE_MS = 100;
const FEATURES = {
  foot_pos: 'Foot positions', foot_vel: 'Foot velocities', hip_vel: 'Hip velocity',
  traj_pos: 'Trajectory positions', traj_dir: 'Trajectory directions', loco_tag: 'Empty-handed or carrying tag',
  box_pos: 'Box position', box_dir: 'Box direction', box_tag: 'Pick or place tag', box_hold: 'Hold state',
};
const QUERY_NAME = { loco: 'Locomotion', pick: 'Pick up', place: 'Put down' };
const MODE_NAME = { idle: 'idle', planner: 'planner mode', pose: 'POSE', stopped: 'stopped' };

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);
const toThreeQuat = (w) => new THREE.Quaternion(w[1], w[2], w[3], w[0]);

const state = {
  db: DBS.includes(params.get('db')) ? params.get('db') : 'loco',
  loco: null,
  filter: params.get('q') ?? '',
  motions: [], shown: [], sel: -1,
  span: null, t: 0, playing: true, speed: 1,
};

$('variant').textContent = `Data variant: ${VARIANTS[VARIANT].label}`;
// UI 3 (the map server's port) is the only page linking here, and from another port the referrer is its origin only.
if (document.referrer && new URL(document.referrer).origin !== location.origin) {
  $('back').href = new URL('/ui3', document.referrer).href;
  $('back-row').hidden = false;
}

async function getLive() {
  try {
    const r = await fetch('api/live', { cache: 'no-store' });
    return r.ok ? await r.json() : null;
  } catch { return null; }
}
const live0 = await getLive();
const bridgeLoco = live0 && live0.loco ? live0.loco : 'emmboth';

const scene = new Scene($('c'));
const cfg = await (await fetch(`${VARIANTS[VARIANT].data}/config.json`)).json();
BOX.size = cfg.data_box_size.slice();
await scene.loadRobot('data');
scene.ghost.root.visible = false;
scene.showGizmos = false;
scene.gizmos.visible = false;
scene.setBoxSize();
const box = new THREE.Group();
const boxMesh = new THREE.Mesh(scene.boxGeom, scene.boxMat);
boxMesh.castShadow = boxMesh.receiveShadow = true;
box.add(boxMesh, new THREE.LineSegments(scene.edgeGeom, new THREE.LineBasicMaterial({ color: '#7a5a34' })));
scene.scene.add(box);

$('loading').textContent = 'Loading the motion database…';
const mm = new MMBackend(cfg, await loadMMData(VARIANTS[VARIANT].data, cfg, VARIANTS[VARIANT].shared));
const L = cfg.pose_layout;
if (!cfg.emm_loco) $('loco').disabled = true;
const panel = new StatePanel($('clipview'), { cfg, backend: mm, bodies: scene.meta.bodies });
await setLoco(bridgeLoco, false);
// The live panel's clips are those of the bridge's Locomotion database, whatever the browser below shows.
const livePanel = new StatePanel($('live'), { cfg, backend: mm, bodies: scene.meta.bodies });
const wantLoco = ['nsm', 'emmboth'].includes(params.get('loco')) ? params.get('loco') : bridgeLoco;
if (wantLoco !== bridgeLoco) await setLoco(wantLoco, false);
$('loading').hidden = true;

// ---- the motions of a database

function rangesOf(d, kindOf) {
  const out = [];
  for (let r = 0; r < d.ranges.length / 2; r++) {
    const a = d.ranges[2 * r], b = d.ranges[2 * r + 1];
    out.push({ frames: d.idx.subarray(a, b), kind: kindOf(r, d.idx[a]), range: r });
  }
  return out;
}

function buildMotions() {
  const W = mm.width, H = L.held[0];
  let list;
  if (state.db === 'loco') list = rangesOf(mm.db.loco, (r, g) => (mm.pose[g * W + H] > 0.5 ? 'carry' : 'loco'));
  else if (state.db === 'clips') {
    list = panel.clips.map((c, i) => ({ frames: Int32Array.from({ length: c.stop - c.start }, (_, k) => c.start + k), kind: 'clip', clipIndex: i }));
  } else {
    const kinds = cfg.controllers.box.range_kind, want = state.db === 'pick' ? 0 : 1;
    list = rangesOf(mm.db.box, (r) => (kinds[r] ? 'place' : 'pick')).filter((m) => kinds[m.range] === want);
  }
  for (const m of list) {
    m.clipIndex ??= panel.clipOf(m.frames[0]);
    const c = panel.clips[m.clipIndex];
    m.clip = c;
    m.at = m.frames[0] - c.start;
  }
  state.motions = list;
}

function applyFilter() {
  const f = state.filter.trim().toLowerCase();
  state.shown = state.motions.map((m, i) => i).filter((i) => !f || state.motions[i].clip.name.toLowerCase().includes(f));
}

function renderList() {
  const n = state.motions.length, s = state.shown.length;
  const frames = state.motions.reduce((t, m) => t + m.frames.length, 0);
  const what = { loco: 'ranges', pick: 'pick ranges', place: 'place ranges', clips: 'clips' }[state.db];
  $('list-note').textContent = `${n} ${n === 1 ? what.slice(0, -1) : what}, ${(frames / FPS / 60).toFixed(1)} min at 60 fps` + (s < n ? `. ${s} match the filter.` : '.');
  const rows = state.shown.map((i) => {
    const m = state.motions[i], c = m.clip;
    const name = c.name.replace(/^ApproachPickV\d_/, '');
    return `<tr data-i="${i}" class="${i === state.sel ? 'is-a' : ''}"><td class="num">${i + 1}</td>` +
      `<td class="name" title="${esc(c.name)}${c.mirror ? ', mirrored' : ''}">${esc(name)}${c.mirror ? ' <span class="muted">mirrored</span>' : ''}</td>` +
      `<td title="${esc(KIND_LABEL[m.kind])}"><span class="key k-${m.kind}"></span></td>` +
      `<td class="num">${(m.at / FPS).toFixed(2)}</td><td class="num">${(m.frames.length / FPS).toFixed(2)}</td></tr>`;
  });
  $('list').innerHTML = '<thead><tr><th>#</th><th>Clip</th><th>Kind</th><th title="Where the motion starts in its clip">From (s)</th><th>Length (s)</th></tr></thead>' +
    `<tbody>${rows.join('')}</tbody>`;
}

function markSelected() {
  for (const tr of $('list').querySelectorAll('tbody tr')) tr.classList.toggle('is-a', +tr.dataset.i === state.sel);
  const tr = $('list').querySelector(`tr[data-i="${state.sel}"]`);
  tr?.scrollIntoView({ block: 'nearest' });
}

// ---- playback

// Root positions and rotations of every frame of a span, from the origin facing +X, as MMBackend.currentPose moves it.
function integrate(frames) {
  const W = mm.width, D = L.root_delta[0], n = frames.length;
  const pos = new Float64Array(n * 3), rot = new Float64Array(n * 4);  // (n, 3), (n, 4)
  let p = [0, 0, 0], r = q.id();
  for (let k = 0; k < n; k++) {
    pos.set(p, k * 3);
    rot.set(r, k * 4);
    const o = frames[k] * W + D;
    p = v3.add(p, q.mulVec(r, [mm.pose[o], mm.pose[o + 1], 0]));
    r = q.norm(q.mul(q.fromYaw(mm.pose[o + 2]), r));
  }
  return { pos, rot };
}

function select(i, { frame = 0, keepTime = false } = {}) {
  if (i < 0 || i >= state.motions.length) return;
  state.sel = i;
  const m = state.motions[i];
  state.span = { m, ...integrate(m.frames) };
  state.t = keepTime ? Math.min(state.t, m.frames.length - 1) : frame;
  $('time').max = String(Math.max(0, m.frames.length - 1));
  panel.show(m.clipIndex, true);
  outlineSpan();
  markSelected();
  syncUrl();
  const c = m.clip;
  $('hud-state').textContent = KIND_LABEL[m.kind];
  $('hud-state').dataset.kind = m.kind;
  $('hud-sub').textContent = `${c.name}${c.mirror ? ', mirrored' : ''}`;
  const h = pose(0).hipPos;
  scene.controls.target.set(h[0], h[1], 0.65);
  scene.camera.position.set(h[0] - 2.4, h[1] - 2.9, 1.75);
}

function outlineSpan() {
  const svg = panel.svg, lay = panel.layout, m = state.span?.m;
  svg.querySelector('.span')?.remove();
  if (!m || !lay) return;
  const a = lay.X(m.at), b = lay.X(m.at + m.frames.length - 1);
  const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  rect.setAttribute('class', 'span');
  rect.setAttribute('x', a.toFixed(1));
  rect.setAttribute('y', '10');
  rect.setAttribute('width', Math.max(2, b - a).toFixed(1));
  rect.setAttribute('height', String(lay.rows.stepLane[0] + lay.rows.stepLane[1] - 10));
  rect.setAttribute('rx', '4');
  svg.insertBefore(rect, svg.firstChild);
}

function rowAt(k) {
  const W = mm.width, g = state.span.m.frames[k];
  return mm.pose.subarray(g * W, (g + 1) * W);
}

// The pose at fractional frame t of the span, blended between its two frames.
function pose(t) {
  const sp = state.span, n = sp.m.frames.length;
  const k = Math.max(0, Math.min(n - 1, Math.floor(t))), k1 = Math.min(n - 1, k + 1), w = Math.min(1, Math.max(0, t - k));
  const a = rowAt(k), b = rowAt(k1);
  const vec = (r, key, m) => Array.from(r.subarray(L[key][0], L[key][0] + m));
  const lerp3 = (key) => v3.lerp(vec(a, key, 3), vec(b, key, 3), w);
  const quat = (key) => q.nlerpShortest(q.norm(vec(a, key, 4)), q.norm(vec(b, key, 4)), w);
  const rootPos = v3.lerp(Array.from(sp.pos.subarray(k * 3, k * 3 + 3)), Array.from(sp.pos.subarray(k1 * 3, k1 * 3 + 3)), w);
  const rootRot = q.nlerpShortest(Array.from(sp.rot.subarray(k * 4, k * 4 + 4)), Array.from(sp.rot.subarray(k1 * 4, k1 * 4 + 4)), w);
  const joints = new Float64Array(29);  // (29,)
  for (let j = 0; j < 29; j++) joints[j] = a[L.joints[0] + j] + (b[L.joints[0] + j] - a[L.joints[0] + j]) * w;
  const hipPos = v3.add(rootPos, q.mulVec(rootRot, lerp3('hip_pos')));
  const hipRot = q.mul(rootRot, quat('hip_quat'));
  const held = a[L.held[0]] + (b[L.held[0]] - a[L.held[0]]) * w;
  const boxPos = v3.add(rootPos, q.mulVec(rootRot, lerp3('box_pos')));
  const boxRot = q.norm(q.mul(rootRot, quat('box_quat')));
  return { hipPos, hipRot, joints, held, boxPos, boxRot, frame: sp.m.frames[k] };
}

// The EMM locomotion takes have no box.
const hasBox = (m) => !m.clip.name.startsWith('emm__');

function draw(dt) {
  if (!state.span) return;
  const p = pose(state.t);
  scene.setPose(p);
  box.visible = $('show-box').checked && hasBox(state.span.m) && p.boxPos.every(Number.isFinite);
  if (box.visible) {
    box.position.set(...p.boxPos);
    box.quaternion.copy(toThreeQuat(p.boxRot));
  }
  scene.follow = $('follow').checked;
  scene.followCharacter(p.hipPos, dt);
  panel.update(null, p.frame);
  const n = state.span.m.frames.length, k = Math.round(state.t);
  $('time').value = String(Math.min(n - 1, k));
  $('time-out').textContent = `${(state.t / FPS).toFixed(2)} / ${((n - 1) / FPS).toFixed(2)} s · frame ${Math.min(n - 1, k) + 1}/${n}` +
    ` · clip ${((state.span.m.at + state.t) / FPS).toFixed(2)} s${p.held > 0.5 ? ' · holding' : ''}`;
}

function advance(dt) {
  if (!state.playing || !state.span) return;
  const n = state.span.m.frames.length;
  state.t += dt * FPS * state.speed;
  if (state.t > n - 1) {
    if ($('autonext').checked) { step(+1); return; }
    state.t = $('loop').checked ? 0 : n - 1;
    if (!$('loop').checked) setPlaying(false);
  }
}

function setPlaying(on) {
  state.playing = on;
  $('play').textContent = on ? 'Pause' : 'Play';
}

function step(dir) {
  const pos = state.shown.indexOf(state.sel);
  const next = state.shown[pos < 0 ? 0 : (pos + dir + state.shown.length) % state.shown.length];
  if (next !== undefined) select(next);
}

// ---- databases and the address

async function setLoco(name, show = true) {
  if (name !== 'nsm' && !mm.locos[name]) {
    $('loading').hidden = false;
    $('loading').textContent = 'Loading the EMM locomotion takes and carry spans…';
    await loadLoco(name, cfg, mm, cfg.emm_loco.dir, EMM_CARRY_DIR);
    $('loading').hidden = true;
  }
  mm.useLoco(name);
  state.loco = name;
  $('loco').value = name;
  panel.setDatabases(false);
  panel.follow = false;
  if (show) {
    setDb(state.db, 0);
    syncUrl();
  }
}

function setDb(name, sel = 0) {
  state.db = name;
  for (const b of document.querySelectorAll('.db-tabs button')) {
    b.classList.toggle('active', b.dataset.db === name);
    b.setAttribute('aria-selected', String(b.dataset.db === name));
  }
  buildMotions();
  applyFilter();
  state.sel = -1;
  renderList();
  select(state.shown.includes(sel) ? sel : (state.shown[0] ?? 0));
}

function syncUrl() {
  const u = new URLSearchParams();
  if (VARIANT !== DEFAULT_VARIANT) u.set('variant', VARIANT);
  if (state.loco !== bridgeLoco) u.set('loco', state.loco);
  u.set('db', state.db);
  if (state.sel >= 0) u.set('m', String(state.sel + 1));
  if (state.filter) u.set('q', state.filter);
  history.replaceState(null, '', `?${u}`);
}

// Click on the clip graph: a frame of the motion seeks there, any other frame plays the whole clip from there.
panel.svg.addEventListener('click', (e) => {
  const lay = panel.layout, m = state.span?.m;
  if (!lay || !m) return;
  const r = panel.svg.getBoundingClientRect();
  const x = (e.clientX - r.left) * (panel.svg.viewBox.baseVal.width / r.width);
  if (x < lay.x0 || x > lay.x1) return;
  const f = Math.round((x - lay.x0) / (lay.x1 - lay.x0) * (lay.n - 1));
  const k = f - m.at;
  if (k >= 0 && k < m.frames.length) { state.t = k; return; }
  const clip = panel.clip;
  $('filter').value = state.filter = '';
  setDb('clips', clip);
  state.t = f;
});

$('loco').addEventListener('change', (e) => setLoco(e.target.value));
for (const b of document.querySelectorAll('.db-tabs button')) b.addEventListener('click', () => { setDb(b.dataset.db); syncUrl(); });
$('filter').value = state.filter;
$('filter').addEventListener('input', (e) => {
  state.filter = e.target.value;
  applyFilter();
  renderList();
  if (!state.shown.includes(state.sel) && state.shown.length) select(state.shown[0]);
  syncUrl();
});
$('list').addEventListener('click', (e) => {
  const tr = e.target.closest('tr[data-i]');
  if (tr) select(+tr.dataset.i);
});
$('play').addEventListener('click', () => setPlaying(!state.playing));
$('prev').addEventListener('click', () => step(-1));
$('next').addEventListener('click', () => step(+1));
$('time').addEventListener('input', (e) => { state.t = +e.target.value; setPlaying(false); });
$('speed').addEventListener('change', (e) => { state.speed = +e.target.value; });
window.addEventListener('keydown', (e) => {
  if (e.target.closest('input[type="search"], select')) return;
  const n = state.span?.m.frames.length ?? 1;
  if (e.key === ' ') { setPlaying(!state.playing); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { setPlaying(false); state.t = Math.max(0, Math.round(state.t) - 1); e.preventDefault(); }
  else if (e.key === 'ArrowRight') { setPlaying(false); state.t = Math.min(n - 1, Math.round(state.t) + 1); e.preventDefault(); }
  else if (e.key === 'ArrowUp') { step(-1); e.preventDefault(); }
  else if (e.key === 'ArrowDown') { step(+1); e.preventDefault(); }
});
new ResizeObserver(() => outlineSpan()).observe(panel.svg.parentElement);

const m0 = Math.max(0, (+params.get('m') || 1) - 1);
setDb(state.db, m0);
window.motions = { state, select, setDb };

// ---- the bridge now

let mtKind = null, mtChosen = false, liveNow = null;
for (const b of document.querySelectorAll('.mmw-kinds button')) {
  b.addEventListener('click', () => { mtKind = b.dataset.kind; mtChosen = true; renderMatching(); });
}

function clipText(c) {
  return c ? `${esc(c.name)}${c.mirror ? ', mirrored' : ''} at ${c.at_s.toFixed(1)} of ${c.len_s.toFixed(1)} s` : 'none';
}

function renderMatching() {
  const m = liveNow && liveNow.match;
  for (const b of document.querySelectorAll('.mmw-kinds button')) {
    b.classList.toggle('active', b.dataset.kind === mtKind);
    b.classList.toggle('inuse', !!m && b.dataset.kind === m.controller.in_use);
  }
  if (!m || !mtKind) { $('mt-weights').innerHTML = ''; $('mt-last').textContent = ''; $('mt-sides').textContent = ''; return; }
  const last = m.last[mtKind], groups = m.groups[mtKind] || [], weights = m.weights[mtKind] || [];
  $('mt-weights').innerHTML = groups.map((g, i) => `<tr><td>${esc(FEATURES[g] || g)}</td><td>${weights[i] ?? ''}</td>` +
    `<td class="share">${last ? `${Math.round(100 * last.share[i])}%` : ''}</td></tr>`).join('');
  const th = mtKind !== 'loco' ? `. Enter below ${m.enter}, leave above ${m.leave}` : '';
  $('mt-last').innerHTML = last ? `Last match: cost ${last.cost.toFixed(2)}, ${clipText(last.clip)}${th}.` : 'No search of this query yet.';
  $('mt-sides').textContent = mtKind !== 'loco' && m.side_costs.length
    ? `The box sides of the last search cost ${m.side_costs.map((c) => c.toFixed(2)).join(', ')}.` : '';
}

function renderLive(lv) {
  liveNow = lv;
  const m = lv && lv.ok ? lv.match : null;
  $('now-status').classList.toggle('bad', !lv);
  $('now-status').textContent = !lv ? 'the bridge does not answer' : !m ? `the bridge: ${MODE_NAME[lv.mode] || lv.mode}, no stream yet` :
    `the bridge: ${MODE_NAME[lv.mode] || lv.mode}${lv.fsm ? `, task state ${lv.fsm}` : ''}`;
  $('play-now').disabled = !m || m.robot.g === null;
  if (!m) { livePanel.update(null, null); $('now-ahead').textContent = ''; renderMatching(); return; }
  const r = m.robot, c = m.controller;
  livePanel.update(stateOf(r.semantics, r.holding), r.g);
  $('now-ahead').textContent = `The highlighted state is the frame the robot plays. The controller runs ${m.ahead_s.toFixed(1)} s ahead of it: ` +
    `${stateOf(c.semantics, c.holding).label}, searching ${QUERY_NAME[c.in_use] || c.in_use}.`;
  if (!mtChosen || !mtKind) mtKind = c.in_use || 'loco';
  renderMatching();
}

$('play-now').addEventListener('click', async () => {
  const g = liveNow && liveNow.match && liveNow.match.robot.g;
  if (g === null || g === undefined) return;
  if (state.loco !== bridgeLoco) await setLoco(bridgeLoco, false);
  const i = livePanel.clipOf(g);
  $('filter').value = state.filter = '';
  setDb('clips', i);
  state.t = g - panel.clips[i].start;
  syncUrl();
  $('c').scrollIntoView({ block: 'center', behavior: 'smooth' });
});

renderLive(live0);
(async function pollLive() {
  for (;;) {
    await new Promise((r) => setTimeout(r, LIVE_MS));
    if (!document.hidden) renderLive(await getLive());
  }
})();

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  advance(dt);
  draw(dt);
  scene.render();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
