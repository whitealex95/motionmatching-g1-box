// Controller.match of lmm-g1-box's page on searches around the SceneBot pick and place (sbplace, Locomotion
// 'emmboth'), for check_parity.py --cases. The page's scripted sequence never makes these searches jump, so the
// rules for a SceneBot clip before its grasp (lmm-g1-box a41bd3f) or its release (1721681) are checked here: the playing frame every 5th
// frame of each SceneBot range and 60 random frames, pick and place queries with and without walkIn, each the
// features of a database frame plus noise.
//
// Run: node tools/page_parity/js_match_cases.mjs <lmm-g1-box checkout> <out.json>

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [repo, out] = process.argv.slice(2);
if (!repo || !out) throw new Error('usage: js_match_cases.mjs <lmm-g1-box> <out.json>');
const { serveWebFetch, webImport } = await import(pathToFileURL(path.join(path.resolve(repo), 'physics/node_env.mjs')).href);
serveWebFetch();
const { Controller, MMBackend, loadMMData, loadLoco } = await webImport('js/controller.js');
const { VARIANTS } = await webImport('js/variants.js');

const variant = 'holosoma-v3-box338x335x360-apv4-scenebox-sbplace-angle60';
const DATA = VARIANTS[variant].data;
const cfg = await (await fetch(`${DATA}/config.json`)).json();
const robot = await (await fetch('data/robot.json')).json();
const b = new MMBackend(cfg, await loadMMData(DATA, cfg, VARIANTS[variant].shared));
const c = new Controller(cfg, b, { bodies: robot.bodies });
await loadLoco('emmboth', cfg, b, cfg.emm_loco.dir, 'data/emm_carry');
c.setLoco('emmboth');

let seed = 12345;
const rand = () => {
  seed = (seed + 0x6D2B79F5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());

const name = 'box', d = b.db[name], n = d.nfeatures, C = b.ctrl[name];
const ranges = [];
for (let r = 0; r < d.ranges.length / 2; r++) ranges.push([d.ranges[2 * r], d.ranges[2 * r + 1], b.rangeClip(name, r).name]);
const frames = [];
for (const [a, e, clip] of ranges) if (clip.startsWith('SceneBot')) for (let f = a; f < e - 20; f += 5) frames.push(f);
for (let i = 0; i < 60; i++) frames.push(Math.floor(rand() * d.idx.length));

const cases = [];
for (const frame of frames) {
  const [a, e] = ranges.find(([a, e]) => frame >= a && frame < e);
  const targets = [a + Math.floor(rand() * 15), Math.floor(rand() * d.idx.length), Math.min(e - 1, frame + 25), Math.max(a, frame - 25)];
  for (const kind of ['pick', 'place']) for (const walkIn of [false, true]) for (const g of targets) {
    const raw = [];
    for (let j = 0; j < n; j++) raw.push((d.X[g * n + j] + 0.1 * gauss()) * C.scale[j] + C.offset[j]);
    b.active = name; b.frame = frame; b.frac = 0; c.walkIn = walkIn;
    const m = c.match(kind, raw);
    cases.push({ frame, kind, walkIn, raw, js: { frame: m.frame, cost: m.cost } });
  }
}
fs.writeFileSync(out, JSON.stringify({ sceneBotPre: Array.from(b.sceneBotPre(name)), beforeEvent: Array.from(b.beforeEventMask(name)), cases }));
console.log(`${cases.length} cases`);
