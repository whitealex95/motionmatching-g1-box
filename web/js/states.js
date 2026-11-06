// The controller states of the HUD and the clips behind the two Matching Databases, drawn under the viewer.

import { locoParts, torsoChain, torsoTilt } from './controller.js';

export const KINDS = {
  loco: 'Locomotion, empty-handed', carry: 'Locomotion, carrying', pick: 'Box, pick tag', place: 'Box, place tag',
};

// semantics of Controller.update, and for Locomotion whether the robot holds a box.
const STATES = {
  locomotion: { loco: 'Locomotion', carry: 'Carrying' },
  navigate: { loco: 'Walking to the box', carry: 'Carrying the box to the marker' },
  'pick-pre': { pick: 'Picking up' }, 'pick-post': { pick: 'Lifting the box' },
  'place-pre': { place: 'Putting down' }, 'place-post': { place: 'Standing up' },
};

export function stateOf(semantics, holding) {
  const s = STATES[semantics];
  if (!s) return { key: semantics, label: semantics, kind: null };
  const kind = semantics === 'locomotion' || semantics === 'navigate' ? (holding ? 'carry' : 'loco') : Object.keys(s)[0];
  return { key: `${semantics}:${kind}`, label: s[kind], kind };
}

// A recorded physical run keeps only the HUD text, which starts with the state's label.
export function stateOfLabel(text) {
  let best = null;
  for (const [semantics, s] of Object.entries(STATES)) {
    for (const [kind, label] of Object.entries(s)) {
      if (text.startsWith(label) && (!best || label.length > best.label.length)) best = { key: `${semantics}:${kind}`, label, kind };
    }
  }
  return best || { key: null, label: text, kind: null };
}

const ROWS = { pelvis: [0.35, 0.85], lean: [0, 100] };
const esc = (s) => s.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);
const clipRows = (clips, first) => clips.rows.map(([n, mirror, start, stop, attach, release]) =>
  ({ name: clips.names[n], mirror: !!mirror, start: first + start, stop: first + stop, attach, release }));

export class StatePanel {
  constructor(root, { cfg, backend, bodies }) {
    this.root = root;
    this.cfg = cfg;
    this.b = backend;
    this.chain = torsoChain(bodies);
    this.chips = Object.fromEntries([...root.querySelectorAll('[data-state]')].map((el) => [el.dataset.state, el]));
    this.svg = root.querySelector('svg');
    this.tip = root.querySelector('.clip-tip');
    this.select = root.querySelector('select');
    this.sampled = new Map();
    this.noEntry = true;
    this.setDatabases(false);
    this.select.addEventListener('change', () => {
      this.follow = this.select.value === 'follow';
      if (!this.follow) this.show(+this.select.value);
    });
    this.follow = true;
    this.clip = null;
    this.frame = null;
    this.active = null;
    this.shownAt = 0;
    new ResizeObserver(() => { if (this.clip !== null) this.show(this.clip, true); }).observe(this.svg.parentElement);
    this.svg.addEventListener('pointermove', (e) => this.hover(e));
    this.svg.addEventListener('pointerleave', () => { this.tip.hidden = true; this.hair?.setAttribute('visibility', 'hidden'); });
    this.show(0);
  }

  // The clips and the Matching Databases in use. With the EMM locomotion takes or the EMM carry spans in use, their clips
  // follow the variant's.
  setDatabases(show = true) {
    const b = this.b, parts = locoParts(b.loco);
    this.clips = clipRows(this.cfg.clips, 0);
    for (const e of [parts.empty === 'emm' ? b.emm : null, parts.carry === 'emm' ? b.emmCarry : null]) {
      if (e) this.clips = this.clips.concat(clipRows(e.cfg.clips, e.first));
    }
    this.markFrames();
    this.showTotals();
    while (this.select.length > 1) this.select.remove(1);
    for (const [i, c] of this.clips.entries()) if (!c.mirror) this.select.add(new Option(c.name, String(i)));
    this.select.value = 'follow';
    this.follow = true;
    this.sampled.clear();
    if (show) this.show(0, true);
  }

  // Per animation frame: 1 empty-handed or 2 carrying in the Locomotion database, 1 pick or 2 place in the Box database.
  markFrames() {
    const b = this.b, n = b.pose.length / b.width, P = b.pose, W = b.width, H = b.L.held[0];
    this.loco = new Uint8Array(n);
    this.box = new Uint8Array(n);
    for (const g of b.db.loco.idx) this.loco[g] = P[g * W + H] > 0.5 ? 2 : 1;
    const d = b.db.box, kinds = this.cfg.controllers.box.range_kind, ne = this.noEntry && b.noEntry ? b.noEntry('box') : null;
    // 3 and 4: pick and place frames that the search does not jump to (MMBackend.noEntry).
    for (let r = 0; r < kinds.length; r++) {
      for (let i = d.ranges[2 * r]; i < d.ranges[2 * r + 1]; i++) this.box[d.idx[i]] = (kinds[r] ? 2 : 1) + (ne && ne[i] ? 2 : 0);
    }
  }

  showTotals() {
    const count = (a, v) => a.reduce((s, x) => s + (x === v), 0);
    const min = (f) => f / 60 / 60;
    const parts = { loco: count(this.loco, 1), carry: count(this.loco, 2), pick: count(this.box, 1) + count(this.box, 3), place: count(this.box, 2) + count(this.box, 4) };
    const bar = (keys) => {
      const total = keys.reduce((s, k) => s + parts[k], 0);
      return `<span class="db-bar">${keys.map((k) => `<span class="k-${k}" style="flex-grow:${parts[k]}" title="${KINDS[k]}: ${min(parts[k]).toFixed(1)} min"></span>`).join('')}</span>` +
        `<span class="db-num">${min(total).toFixed(1)} min: ${keys.map((k) => `${min(parts[k]).toFixed(1)} ${k === 'loco' ? 'empty-handed' : k === 'carry' ? 'carrying' : k}`).join(', ')}</span>`;
    };
    const nsm = new Set(), ap = new Set(), emm = new Set();
    for (const c of this.clips) (c.name.startsWith('ApproachPick') ? ap : c.name.startsWith('emm__') ? emm : nsm).add(c.name);
    const extra = [ap.size && `${ap.size} ApproachPick clips`, emm.size && `${emm.size} EMM locomotion takes`].filter(Boolean);
    this.root.querySelector('.db-totals').innerHTML =
      `<div class="db-row"><span class="db-name">Locomotion</span>${bar(['loco', 'carry'])}</div>` +
      `<div class="db-row"><span class="db-name">Box</span>${bar(['pick', 'place'])}</div>` +
      `<p class="muted small">From ${nsm.size} NSM episodes${extra.length ? ` and ${extra.join(' and ')}` : ''}, each also mirrored, at 60 fps.</p>`;
  }

  clipOf(g) {
    let lo = 0, hi = this.clips.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.clips[mid].start <= g) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  // Torso link tilt from vertical (deg) and pelvis height (m) of an animation frame.
  sample(g) {
    const b = this.b, L = b.L, r0 = g * b.width, P = b.pose, h = r0 + L.hip_quat[0];
    const lean = torsoTilt(this.chain, [P[h], P[h + 1], P[h + 2], P[h + 3]], (d) => P[r0 + L.joints[0] + d]);
    return { pelvis: P[r0 + L.hip_pos[0] + 2], lean };
  }

  show(i, force = false) {
    if (i === this.clip && !force) return;
    this.clip = i;
    const c = this.clips[i], n = c.stop - c.start;
    const w = Math.max(320, this.svg.parentElement.clientWidth), x0 = 88, x1 = w - 8;
    const X = (f) => x0 + (f / Math.max(1, n - 1)) * (x1 - x0);
    const rows = { pelvis: [16, 34], lean: [58, 34], locoLane: [100, 12], boxLane: [118, 12], stepLane: [136, 12] }, axisY = 164, H = 180;
    this.layout = { X, x0, x1, n, rows };
    if (this.sampled.size > 64) this.sampled.clear();
    if (!this.sampled.has(i)) this.sampled.set(i, Array.from({ length: n }, (_, f) => this.sample(c.start + f)));
    this.samples = this.sampled.get(i);
    const line = (key) => {
      const [y, h] = rows[key], [lo, hi] = ROWS[key];
      return this.samples.map((s, f) => `${f ? 'L' : 'M'}${X(f).toFixed(1)},${(y + h - (Math.min(hi, Math.max(lo, s[key])) - lo) / (hi - lo) * h).toFixed(1)}`).join('');
    };
    const lane = (arr, key, names) => {
      const [y, h] = rows[key];
      let out = `<rect class="track" x="${x0}" y="${y}" width="${x1 - x0}" height="${h}" rx="3"/>`;
      for (let a = 0; a < n;) {
        const v = arr[c.start + a];
        let e = a + 1;
        while (e < n && arr[c.start + e] === v) e++;
        if (v) out += `<rect class="k-${names[v - 1]}" x="${X(a).toFixed(1)}" y="${y}" width="${Math.max(1, X(e - 1) - X(a)).toFixed(1)}" height="${h}" rx="3"/>`;
        a = e;
      }
      return out;
    };
    const label = (key, text) => `<text x="0" y="${rows[key][0] + rows[key][1] / 2 + 4}">${text}</text>`;
    let svg = label('pelvis', 'Pelvis height') + label('lean', 'Torso lean') + label('locoLane', 'Locomotion') + label('boxLane', 'Box') +
      label('stepLane', 'Last step');
    for (const key of ['pelvis', 'lean']) {
      const [y, h] = rows[key];
      svg += `<line class="grid" x1="${x0}" x2="${x1}" y1="${y + h}" y2="${y + h}"/><path class="curve" d="${line(key)}"/>`;
    }
    svg += lane(this.loco, 'locoLane', ['loco', 'carry']) + lane(this.box, 'boxLane', ['pick', 'place', 'pick noentry', 'place noentry']);
    const [sy, sh] = rows.stepLane;
    svg += `<rect class="track" x="${x0}" y="${sy}" width="${x1 - x0}" height="${sh}" rx="3"/>`;
    this.steps = this.b.steps.clips[i] ?? [];
    for (const st of this.steps) {
      const a = X(st.step), e = X(Math.max(st.step, st.lift - 1));
      svg += `<rect class="k-${st.kind} planted" x="${a.toFixed(1)}" y="${sy}" width="${Math.max(1, e - a).toFixed(1)}" height="${sh}" rx="3"/>` +
        `<line class="step" x1="${a.toFixed(1)}" x2="${a.toFixed(1)}" y1="10" y2="${sy + sh}"/>`;
    }
    for (const [f, name] of [[c.attach, 'grasp'], [c.release, 'release']]) {
      if (f == null || f < 0 || f >= n) continue;
      svg += `<line class="event" x1="${X(f)}" x2="${X(f)}" y1="10" y2="${sy + sh}"/><text class="event-label" x="${X(f)}" y="8" text-anchor="middle">${name}</text>`;
    }
    const perSecond = (x1 - x0) / (n / 60), step = [1, 2, 5, 10, 20].find((v) => v * perSecond >= 44) || 30;
    for (let s = 0; s * 60 < n; s += step) svg += `<text class="tick" x="${X(s * 60)}" y="${axisY + 10}" text-anchor="middle">${s} s</text>`;
    svg += `<line class="hair" x1="0" x2="0" y1="10" y2="${sy + sh}" visibility="hidden"/>` +
      `<line class="playhead" x1="0" x2="0" y1="10" y2="${sy + sh + 4}" visibility="hidden"/>`;
    this.svg.setAttribute('viewBox', `0 0 ${w} ${H}`);
    this.svg.setAttribute('height', H);
    this.svg.innerHTML = svg;
    this.hair = this.svg.querySelector('.hair');
    this.playhead = this.svg.querySelector('.playhead');
    this.root.querySelector('.clip-name').textContent = `${c.name}${c.mirror ? ', mirrored' : ''} · ${(n / 60).toFixed(1)} s`;
    this.frame = null;
  }

  hover(e) {
    if (!this.layout) return;
    const { x0, x1, n } = this.layout, box = this.svg.getBoundingClientRect();
    const x = (e.clientX - box.left) * (this.svg.viewBox.baseVal.width / box.width);
    if (x < x0 || x > x1) { this.tip.hidden = true; this.hair.setAttribute('visibility', 'hidden'); return; }
    const f = Math.round((x - x0) / (x1 - x0) * (n - 1)), c = this.clips[this.clip], g = c.start + f, s = this.samples[f];
    this.hair.setAttribute('x1', this.layout.X(f));
    this.hair.setAttribute('x2', this.layout.X(f));
    this.hair.setAttribute('visibility', 'visible');
    const bx = this.box[g], dbs = [this.loco[g] && KINDS[this.loco[g] === 1 ? 'loco' : 'carry'],
      bx && KINDS[bx % 2 ? 'pick' : 'place'] + (bx > 2 ? ', played on but not searched' : '')].filter(Boolean);
    const st = this.steps.find((x) => f >= x.step && f < x.lift);
    const step = st ? `<div>Feet down from the last step, ${((st.event - st.step) / 60).toFixed(2)} s before the ${st.kind === 'pick' ? 'grasp' : 'release'}, to the first step after it</div>` : '';
    this.tip.innerHTML = `<div class="t">${(f / 60).toFixed(2)} s</div><div>Pelvis ${s.pelvis.toFixed(3)} m · torso lean ${s.lean.toFixed(0)}°</div>` +
      `<div>${dbs.length ? dbs.map(esc).join('<br>') : 'in no database'}</div>${step}`;
    this.tip.hidden = false;
    const r = this.root.getBoundingClientRect();
    this.tip.style.left = `${Math.min(e.clientX - r.left + 12, r.width - 220)}px`;
    this.tip.style.top = `${e.clientY - r.top + 12}px`;
  }

  // frame: the animation frame the controller plays, or null when it does not run (video, recorded physical run).
  update(state, frame) {
    const key = state && state.kind ? state.key : null;
    if (key !== this.active) {
      this.chips[this.active]?.classList.remove('active');
      this.chips[key]?.classList.add('active');
      this.active = key;
    }
    if (document.body.classList.contains('recording')) return;
    const has = frame !== null && frame !== undefined;
    // Motion Matching may jump between clips several times a second. The graph follows at most twice a second.
    if (has && this.follow) {
      const i = this.clipOf(frame), now = performance.now();
      if (i !== this.clip && now - this.shownAt > 500) {
        this.show(i);
        this.shownAt = now;
      }
    }
    const c = this.clips[this.clip];
    if (!has || frame < c.start || frame >= c.stop) {
      this.playhead?.setAttribute('visibility', 'hidden');
      this.root.querySelector('.clip-at').textContent = '';
      this.frame = null;
      return;
    }
    if (frame === this.frame) return;
    this.frame = frame;
    const x = this.layout.X(frame - c.start);
    this.playhead.setAttribute('x1', x);
    this.playhead.setAttribute('x2', x);
    this.playhead.setAttribute('visibility', 'visible');
    this.root.querySelector('.clip-at').textContent = ` · now at ${((frame - c.start) / 60).toFixed(1)} s`;
  }
}
