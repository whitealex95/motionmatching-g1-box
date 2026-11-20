// The scripted sequence of lmm-g1-box's page run kinematically in node with the page's own modules (sbplace variant,
// the live page's Locomotion 'emmboth'), one JSON line per controller frame (60 Hz). check_parity.py runs mm_g1
// the same way and compares.
//
// --states also writes the whole state of the sequence before each frame (one JSON line per frame), for the
// teacher-forced comparison of check_parity.py --forced.
//
// --foot-land M sets the foot lock's landHeight (the page's 0.02 m). The scripted sequence has no foot that waits at
// 0.02 m, so a small height (0.002) is the way to check that rule: give check_parity.py the same --foot-land.
//
// Run: node tools/page_parity/js_trace.mjs <lmm-g1-box checkout> <out.jsonl> [--frames N] [--states states.jsonl]
//                                           [--foot-land M] [--w-pick W] [--w-place W]
//
// --w-pick W, --w-place W: the weights of the Pick up and the Put down query, comma separated. With 9 values the box
// position and the box direction have a weight for x and one for y (give check_parity.py the same).

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [repo, out] = process.argv.slice(2);
if (!repo || !out) throw new Error('usage: js_trace.mjs <lmm-g1-box> <out.jsonl> [--frames N]');
const fi = process.argv.indexOf('--frames');
const maxFrames = fi > 0 ? +process.argv[fi + 1] : 9000;
const si = process.argv.indexOf('--states');
const statesOut = si > 0 ? process.argv[si + 1] : null;
const li = process.argv.indexOf('--foot-land');
const footLand = li > 0 ? { landHeight: +process.argv[li + 1] } : {};
const { serveWebFetch, webImport } = await import(pathToFileURL(path.join(path.resolve(repo), 'physics/node_env.mjs')).href);
serveWebFetch();
const { Controller, MMBackend, loadMMData, loadLoco } = await webImport('js/mm/controller.js');
const { FootLock } = await webImport('js/mm/ik.js');
const { Sim, BOX } = await webImport('js/mm/sim.js');
const { VARIANTS, DEFAULT_VARIANT } = await webImport('js/mm/variants.js');
const { q } = await webImport('js/mm/math.js');

const variant = DEFAULT_VARIANT;
const DATA = VARIANTS[variant].data;
const cfg = await (await fetch(`${DATA}/config.json`)).json();
BOX.size = [cfg.data_box_size[1], cfg.data_box_size[1], cfg.data_box_size[2]];
const robot = await (await fetch('data/robot.json')).json();
const backend = new MMBackend(cfg, await loadMMData(DATA, cfg, VARIANTS[variant].shared));
const controller = new Controller(cfg, backend, { bodies: robot.bodies });
await loadLoco('emmboth', cfg, backend, cfg.emm_loco.dir, 'data/emm_carry');
controller.setLoco('emmboth');
for (const kind of ['pick', 'place']) {
  const wi = process.argv.indexOf(`--w-${kind}`);
  if (wi > 0) controller.setWeights(kind, process.argv[wi + 1].split(',').map(Number));
}
const sim = new Sim(controller, new FootLock(robot.bodies, footLand), cfg);
sim.startScripted();

// The controller's state that the next frame depends on, and what the page shows of it (navSide to fsm).
const C_KEYS = ['mode', 'boxTag', 'holding', 'sim', 'desiredRot', 'desiredVel', 'searchTimer', 'forceSearch', 'risesFrom', 'riseWalkIn',
  'commit', 'target', 'navSide', 'stance', 'boxCost', 'calledOff', 'engaged', 'reapproach', 'leaveBox', 'missed', 'walkIn',
  'navigating', 'arriving', 'trajectory', 'curr', 'pose', 'off', 'tsrc', 'tdst', 'events', 'semantics', 'searchAt', 'matchAt',
  'finishing', 'clock', 'lastEvent', 'fsm'];
const r = (v, n = 6) => +v.toFixed(n);
const pick = (o, keys) => Object.fromEntries(keys.map((k) => [k, o[k]]));
const stateOf = () => {
  const c = sim.controller, b = c.backend, w = sim.world;
  return {
    c: pick(c, C_KEYS),
    b: pick(b, ['active', 'frame', 'frac', 'loco']),
    w: pick(w, ['boxes', 'tables', 'nextId', 'pickId', 'holdingId', 'placeTarget', 'regrasp']),
    s: pick(sim, ['scenario', 'simTime', 'stats', 'forceSearch', 'grip', 'parked', 'heldPose', 'renderJoints', 'supports']),
    f: { feet: sim.footLock.feet, skate: sim.footLock.skate, waits: sim.footLock.waits },
  };
};
const plain = (k, v) => (ArrayBuffer.isView(v) ? Array.from(v) : v === Infinity ? 'inf' : v);
const states = [];
const lines = [];
let tail = 90;
for (let k = 0; k < maxFrames; k++) {
  if (statesOut) states.push(JSON.stringify(stateOf(), plain));
  sim.step();
  const c = sim.controller, b = c.backend, P = c.pose, w = sim.world;
  lines.push(JSON.stringify({
    k, step: sim.scenario.index, mode: c.mode, tag: c.boxTag, db: b.active, frame: b.frame, row: b.db[b.active].idx[b.frame],
    root: P.rootPos.map((v) => r(v)), yaw: r(q.yaw(P.rootRot)), hip: P.hipPos.map((v) => r(v)), held: r(P.held),
    joints: Array.from(sim.renderJoints).map((v) => r(v, 5)), holding: w.holdingId, pick: w.pickId,
    boxes: w.boxes.map((x) => [x.id, r(x.pos[0]), r(x.pos[1]), r(x.yaw)]),
    place: w.placeTarget ? [r(w.placeTarget.pos[0]), r(w.placeTarget.pos[1]), r(w.placeTarget.yaw)] : null,
    target: c.target ? c.target.side : null, cost: Number.isFinite(c.boxCost) ? r(c.boxCost, 4) : null,
    fsm: c.fsm.state, play: c.playCost(),
  }));
  if (sim.done && --tail <= 0) break;
}
fs.writeFileSync(out, lines.join('\n') + '\n');
if (statesOut) {
  states.push(JSON.stringify(stateOf(), plain));
  fs.writeFileSync(statesOut, states.join('\n') + '\n');
}
console.log(`${lines.length} frames, done=${sim.done}, failed=${JSON.stringify(sim.failed)}, step ${sim.scenario.index}`);
