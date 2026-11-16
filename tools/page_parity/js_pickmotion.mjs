// The page's own web/js/pickmotion.js run in node on the default variant, for a few boxes and table heights. It
// writes the solved rows of the Animation Database (float32, one file per case) and the info of each solve.
// check_pickmotion.py runs mm_g1/pickmotion.py on the same cases and compares.
//
// Run: node tools/page_parity/js_pickmotion.mjs <lmm-g1-box checkout> <out dir>

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [repo, out] = process.argv.slice(2);
if (!repo || !out) throw new Error('usage: js_pickmotion.mjs <lmm-g1-box> <out dir>');
const REPO = path.resolve(repo);
const { serveWebFetch, webImport, loadPhysics } = await import(pathToFileURL(path.join(REPO, 'physics/node_env.mjs')).href);
serveWebFetch();
const { MMBackend, loadMMData } = await webImport('js/controller.js');
const { VARIANTS, DEFAULT_VARIANT } = await webImport('js/variants.js');
const { BodyModel } = await webImport('js/edit/bodyik.js');
const { PickMotion, MANUAL } = await webImport('js/pickmotion.js');

const V = VARIANTS[DEFAULT_VARIANT];
const cfg = await (await fetch(`${V.data}/config.json`)).json();
const backend = new MMBackend(cfg, await loadMMData(V.data, cfg, V.shared));
const P = await loadPhysics();
const model = new BodyModel(P.mj, P.xml, P.meta);
const pm = new PickMotion(cfg, backend, model);

const DATA_BOX = cfg.box_size.slice();
const SIZE = [0.30, 0.38, 0.30];
// v4.0.1's box of the list standing 20 in high, 14 in between the palms (12 x 14 x 20 in).
const TALL = [12, 14, 20].map((v) => +(v * 0.0254).toFixed(4));
const CASES = [
  { name: 'floor', autoHeight: true, autoSize: false, support: 0, box: DATA_BOX },
  { name: 'support0.3', autoHeight: true, autoSize: false, support: 0.3, box: DATA_BOX },
  { name: 'support0.5', autoHeight: true, autoSize: false, support: 0.5, box: DATA_BOX },
  { name: 'support0.7', autoHeight: true, autoSize: false, support: 0.7, box: DATA_BOX },
  { name: 'size30x38x30', autoHeight: true, autoSize: true, support: 0, box: SIZE },
  { name: 'size30x38x30+0.5', autoHeight: true, autoSize: true, support: 0.5, box: SIZE },
  { name: 'size12x14x20in', autoHeight: true, autoSize: true, support: 0, box: TALL },
  // The manual palm shift on the side faces and the grip kept on the held box.
  { name: 'grasp+grip', autoHeight: true, autoSize: false, support: 0.5, box: DATA_BOX,
    manual: { grasp: [[0.02, 0.01], [0.02, 0.01]], gripKeep: true } },
];

fs.mkdirSync(out, { recursive: true });
const W = backend.width;
const clips = pm.clips.map((c) => ({ kind: c.kind, row: c.row, mirror: c.mirror, start: c.start, stop: c.stop, attach: c.attach, release: c.release }));
// The rows the data has, so both sides can check the floor case against them.
const dataBin = Buffer.alloc(clips.reduce((n, c) => n + (c.stop - c.start) * W * 4, 0));
let off = 0;
for (const c of clips) {
  const a = backend.pose.subarray(c.start * W, c.stop * W);
  Buffer.from(a.buffer, a.byteOffset, a.byteLength).copy(dataBin, off);
  off += a.byteLength;
}
fs.writeFileSync(path.join(out, 'data.bin'), dataBin);

// Rows of the Animation Database where lift() is read: the clip's start, its grasp and its last frame.
const liftRows = clips.flatMap((c) => [c.start, c.start + 40, c.start + Math.max(0, c.attach), c.stop - 1].filter((g) => g >= c.start && g < c.stop));

const slim = (r) => (r === null ? null : {
  failed: r.failed ?? null, pelvisMin: r.pelvisMin, basePelvisMin: r.basePelvisMin,
  palmMiss: r.palmMiss ?? null, offset: r.offset, ms: r.ms, clipRow: r.clip.row,
  edit: { forward: r.edit.forward, left: r.edit.left, up: r.edit.up, yaw: r.edit.yaw, box: r.edit.box, alignBox: r.edit.alignBox, fitBox: r.edit.fitBox, gripKeep: r.edit.grip.keep },
  solved: r.solved === null ? null : {
    frames: r.solved.qpos.length, attach: r.solved.attach, stop: r.solved.stop, lifted: r.solved.lifted,
    boxSize: r.solved.boxSize, physBox: r.solved.physBox, grip: r.solved.grip, feetLeft: r.solved.feetLeft,
    stand: r.solved.stand, palm0: r.solved.palm0,
  },
});

const cases = [];
for (const [i, cs] of CASES.entries()) {
  pm.restore();
  pm.settings.autoHeight = cs.autoHeight;
  pm.settings.autoSize = cs.autoSize;
  Object.assign(pm.manual, { ...MANUAL, grasp: MANUAL.grasp.map((g) => g.slice()) }, cs.manual ?? {});
  pm.clearCache();
  pm.solveMs = 0;
  const changed = pm.sync({ pick: cs.support, place: cs.support }, cs.box);
  const bin = Buffer.alloc(dataBin.length);
  let o = 0;
  for (const c of clips) {
    const a = backend.pose.subarray(c.start * W, c.stop * W);
    Buffer.from(a.buffer, a.byteOffset, a.byteLength).copy(bin, o);
    o += a.byteLength;
  }
  const file = `case${i}.bin`;
  fs.writeFileSync(path.join(out, file), bin);
  cases.push({
    ...cs, manual: cs.manual ?? null, bin: file, changed, offsets: pm.offsets(), solveMs: pm.solveMs,
    info: { pick: slim(pm.info.pick), place: slim(pm.info.place) },
    lift: Object.fromEntries(liftRows.map((g) => [g, pm.lift(g)])),
  });
  console.log(`${cs.name}: ${changed.join(',')} ${pm.solveMs.toFixed(0)} ms`);
}
pm.restore();

fs.writeFileSync(path.join(out, 'info.json'), JSON.stringify({
  variant: DEFAULT_VARIANT, node: process.version, mujoco: P.mj.mj_versionString ? P.mj.mj_versionString() : String(P.mj.mj_version()),
  width: W, clips, liftRows, cases,
}, null, 1));
console.log(`wrote ${out}`);
process.exit(0);
