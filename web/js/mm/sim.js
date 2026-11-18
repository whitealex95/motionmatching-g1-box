// The kinematic side without the page: the boxes, the scripted sequence and the controller step with the
// grasp and the release. The page (pages/index.js) and the headless evaluation (tools/eval_physical.mjs) both drive it.

import { SCENARIO, INITIAL_BOXES, INITIAL_TABLES } from './scenario.js';
import { q, v3, wrapAngle } from './math.js';

// The box of the holosoma retargets: X depth away from the carrier, Y between the palms, Z height.
// The page sets it from the data variant's config.
export const BOX = { size: [0.30, 0.38, 0.30] };

// A table: [depth along its x axis, width along its y axis, height] (m), and how far inside its edge a box put there
// stands. A box on a table is picked up from outside the table only: from a side whose stance, STANCE_OUT from the
// box centre, is off the table.
export const TABLE = { size: [0.6, 0.9, 0.5], inset: 0.02, top: 0.03 };
// The robot takes a box on a table from a side whose stance, STANCE_OUT from the box, is STANCE_CLEAR off the table.
const STANCE_OUT = 0.45;
const STANCE_CLEAR = 0.15;

export const PLAY_BOXES = [
  { label: 'A', pos: [1.8, 0.5], yaw: 0.2 },
  { label: 'B', pos: [-1.6, 1.8], yaw: 0.9 },
  { label: 'C', pos: [0.8, -2.0], yaw: -0.4 },
  { label: 'D', table: 'T1', edge: 2 },
];
const PLAY_TABLES = [{ label: 'T1', pos: [-1.2, -2.2], yaw: 0.3 }];

const smoothstep = (x) => { const t = Math.min(1, Math.max(0, x)); return t * t * (3 - 2 * t); };

// Copies objects, arrays, typed arrays, maps and class instances, once each, so references that the state
// shares stay shared in the copy. Objects in keep and typed arrays of 4096 or more values (databases) are not copied.
function copyState(root, keep) {
  const memo = new Map();
  const copy = (v) => {
    if (v === null || typeof v !== 'object' || keep.has(v)) return v;
    if (memo.has(v)) return memo.get(v);
    let out;
    if (ArrayBuffer.isView(v)) out = v.length >= 4096 ? v : v.slice();
    else if (Array.isArray(v)) out = new Array(v.length);
    else if (v instanceof Map) out = new Map();
    else if (v instanceof Set) out = new Set();
    else out = Object.create(Object.getPrototypeOf(v));
    memo.set(v, out);
    if (ArrayBuffer.isView(v)) return out;
    if (v instanceof Map) for (const [k, x] of v) out.set(k, copy(x));
    else if (v instanceof Set) for (const x of v) out.add(copy(x));
    else for (const k of Object.keys(v)) out[k] = copy(v[k]);
    return out;
  };
  return copy(root);
}

const SIM_STATE = ['controller', 'footLock', 'world', 'scenario', 'simTime', 'stats', 'forceSearch', 'grip', 'parked', 'heldPose', 'renderJoints', 'supports'];

// Boxes carry a revision that counts the moves made by commands (the scripted sequence, the player). The
// physical mode places a box in the physics only when its revision changes, so the moves that follow the
// physics (the grasp, the release and the boxes fed back from the physics) never teleport a box.
// A box rests on the floor or on a table: z is the height of what it stands on, table the id of that table or null.
// Tables carry one revision for all of them, which the physical mode follows.
class World {
  constructor() { this.reset([]); }
  reset(boxes, tables = []) {
    this.boxes = [];
    this.tables = [];
    this.nextId = 1;
    this.nextTableId = 1;
    this.tablesRev = 0;
    this.pickId = null;
    this.holdingId = null;
    this.placeTarget = null;
    this.selectedId = null;
    this.selectedTable = null;
    this.walkTo = null;  // {pos: [x, y], yaw: null or the facing on arrival}, a click on the floor with empty hands
    this.regrasp = null;
    for (const t of tables) this.addTable(t.label, t.pos, t.yaw, t.size);
    for (const b of boxes) {
      if (b.table) this.boxOnTable(this.addBox(b.label, [0, 0], 0), b.table, b.edge ?? 2, b.along ?? 0);
      else this.addBox(b.label, b.pos, b.yaw);
    }
  }
  find(key) { return this.boxes.find((b) => b.id === key || b.label === key) || null; }
  floorBoxes() { return this.boxes.filter((b) => b.id !== this.holdingId); }
  // The boxes that stand on the floor, which Locomotion steers around. A table covers the boxes on it.
  groundBoxes() { return this.floorBoxes().filter((b) => !b.table); }
  addBox(label, pos, yaw = 0) {
    const used = new Set(this.boxes.map((b) => b.label));
    if (!label || used.has(label)) {
      label = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').find((c) => !used.has(c)) || String(this.nextId);
    }
    const b = { id: this.nextId++, label, pos: pos.slice(0, 2), yaw, rev: 0, z: 0, table: null };
    this.rest(b);
    this.boxes.push(b);
    return b;
  }
  removeBox(key) {
    const b = this.find(key);
    if (!b) return;
    this.boxes = this.boxes.filter((x) => x !== b);
    if (this.pickId === b.id) this.pickId = null;
    if (this.holdingId === b.id) { this.holdingId = null; this.placeTarget = null; }
    if (this.selectedId === b.id) this.selectedId = null;
    if (this.regrasp === b.id) this.regrasp = null;
  }
  moveBox(key, pos, yaw) {
    const b = this.find(key);
    if (!b || b.id === this.holdingId) return;
    b.pos = pos.slice(0, 2);
    if (yaw !== undefined) b.yaw = yaw;
    this.rest(b);
    b.rev++;
  }
  rotateBox(key, d) { const b = this.find(key); if (b && b.id !== this.holdingId) { b.yaw += d; b.rev++; } }

  // ---- tables

  findTable(key) { return this.tables.find((t) => t.id === key || t.label === key) || null; }
  addTable(label, pos, yaw = 0, size = TABLE.size) {
    const used = new Set(this.tables.map((t) => t.label));
    if (!label || used.has(label)) { let k = 1; while (used.has(`T${k}`)) k++; label = `T${k}`; }
    const t = { id: `t${this.nextTableId++}`, label, pos: pos.slice(0, 2), yaw, size: size.slice() };
    this.tables.push(t);
    this.tablesRev++;
    return t;
  }
  // The boxes on a table move and turn with it, and come down to the floor when it goes.
  moveTable(key, pos, yaw) {
    const t = this.findTable(key);
    if (!t) return;
    const turn = yaw === undefined ? 0 : yaw - t.yaw, c = Math.cos(turn), s = Math.sin(turn);
    for (const b of this.boxes) {
      if (b.table !== t.id || b.id === this.holdingId) continue;
      const dx = b.pos[0] - t.pos[0], dy = b.pos[1] - t.pos[1];
      b.pos = [pos[0] + c * dx - s * dy, pos[1] + s * dx + c * dy];
      b.yaw += turn;
      b.rev++;
    }
    if (this.placeTarget && this.placeTarget.table === t.id) {
      const p = this.placeTarget, dx = p.pos[0] - t.pos[0], dy = p.pos[1] - t.pos[1];
      this.placeTarget = { ...p, pos: [pos[0] + c * dx - s * dy, pos[1] + s * dx + c * dy], yaw: p.yaw + turn };
    }
    t.pos = pos.slice(0, 2);
    if (yaw !== undefined) t.yaw = yaw;
    this.tablesRev++;
  }
  resizeTable(key, size) {
    const t = this.findTable(key);
    if (!t) return;
    t.size = size.slice();
    this.tablesRev++;
    for (const b of this.boxes) if (b.table === t.id && b.id !== this.holdingId) { this.rest(b); b.rev++; }
    if (this.placeTarget && this.placeTarget.table === t.id) this.placeTarget = this.targetAt(this.placeTarget.pos, this.placeTarget.yaw);
  }
  removeTable(key) {
    const t = this.findTable(key);
    if (!t) return;
    this.tables = this.tables.filter((x) => x !== t);
    if (this.selectedTable === t.id) this.selectedTable = null;
    this.tablesRev++;
    for (const b of this.boxes) if (b.table === t.id && b.id !== this.holdingId) { this.rest(b); b.rev++; }
    if (this.placeTarget && this.placeTarget.table === t.id) this.placeTarget = this.targetAt(this.placeTarget.pos, this.placeTarget.yaw);
  }
  // The point in the frame of table t, and whether it is on the table's top.
  tableLocal(t, p) {
    const c = Math.cos(t.yaw), s = Math.sin(t.yaw), dx = p[0] - t.pos[0], dy = p[1] - t.pos[1];
    return [c * dx + s * dy, -s * dx + c * dy];
  }
  onTable(t, p, margin = 0) {
    const l = this.tableLocal(t, p);
    return Math.abs(l[0]) <= t.size[0] / 2 + margin && Math.abs(l[1]) <= t.size[1] / 2 + margin;
  }
  // What a box centred at p stands on: { z, table } of the highest table under it, or the floor.
  support(p) {
    let out = { z: 0, table: null };
    for (const t of this.tables) if (this.onTable(t, p) && t.size[2] > out.z) out = { z: t.size[2], table: t.id };
    return out;
  }
  rest(b) { const s = this.support(b.pos); b.z = s.z; b.table = s.table; }
  // The pose of a box at an edge of table t: edge 0 is the table's +x side, 1 its +y side, 2 its -x side, 3 its -y side.
  // The box X axis points into the table, away from a robot standing outside the edge. along: from the middle of the
  // edge, to the left of a robot facing the table (m).
  edgePose(t, edge, along = 0, boxSize = BOX.size) {
    const out = edge * Math.PI / 2, nx = Math.cos(t.yaw + out), ny = Math.sin(t.yaw + out);
    const reach = (edge % 2 === 0 ? t.size[0] : t.size[1]) / 2 - TABLE.inset - boxSize[0] / 2;
    const lx = -ny, ly = nx;
    return { pos: [t.pos[0] + nx * reach - lx * along, t.pos[1] + ny * reach - ly * along], yaw: t.yaw + out + Math.PI };
  }
  boxOnTable(b, tableKey, edge = 2, along = 0) {
    const t = this.findTable(tableKey);
    if (!t || !b || b.id === this.holdingId) return false;
    const p = this.edgePose(t, edge, along);
    b.pos = p.pos;
    b.yaw = p.yaw;
    b.z = t.size[2];
    b.table = t.id;
    b.rev++;
    return true;
  }
  // The edge of table t nearest to the point p: { edge, along } as edgePose takes them.
  nearestEdge(t, p) {
    const l = this.tableLocal(t, p);
    const inside = [t.size[0] / 2 - l[0], t.size[1] / 2 - l[1], t.size[0] / 2 + l[0], t.size[1] / 2 + l[1]];
    const edge = inside.reduce((k, g, i) => (g < inside[k] ? i : k), 0);
    return { edge, along: [-l[1], l[0], l[1], -l[0]][edge] };
  }
  // The sides (indices of targetSides) a robot may take a target {pos, yaw, table} from, or null for every side. When no
  // stance is clear of the table, the one farthest off it.
  sidesFor(target, n) {
    const t = target && target.table ? this.findTable(target.table) : null;
    if (!t) return null;
    const off = Array.from({ length: n }, (_, k) => {
      const y = target.yaw + k * 2 * Math.PI / n, l = this.tableLocal(t, [target.pos[0] - STANCE_OUT * Math.cos(y), target.pos[1] - STANCE_OUT * Math.sin(y)]);
      return Math.max(Math.abs(l[0]) - t.size[0] / 2, Math.abs(l[1]) - t.size[1] / 2);
    });
    const out = off.map((d, k) => (d > STANCE_CLEAR ? k : -1)).filter((k) => k >= 0);
    return out.length ? out : [off.indexOf(Math.max(...off))];
  }

  setPick(key) {
    const b = key === null ? null : this.find(key);
    if (this.holdingId !== null) return false;
    this.pickId = b ? b.id : null;
    this.walkTo = null;
    return true;
  }
  targetAt(pos, yaw) { const s = this.support(pos); return { pos: pos.slice(0, 2), yaw, z: s.z, table: s.table }; }
  setPlace(pos, yaw) {
    if (this.holdingId === null) return false;
    this.placeTarget = this.targetAt(pos, yaw);
    this.walkTo = null;
    return true;
  }
  // The held box goes down at an edge of a table, as a box put there by boxOnTable.
  setPlaceOnTable(tableKey, edge = 2, along = 0) {
    const t = this.findTable(tableKey);
    if (!t) return false;
    const p = this.edgePose(t, edge, along);
    return this.setPlace(p.pos, p.yaw);
  }
  clearTargets() { this.pickId = null; this.placeTarget = null; this.regrasp = null; }
  pickBox() { return this.pickId === null ? null : this.find(this.pickId); }
  // The physical box left the hands (physical mode): the box is where the physics has it. retry: it stays the box to
  // pick up, and a place target stays for after the new grasp. Otherwise no target is left. z: the height of the
  // box's bottom in the physics, which tells a box on a table from one that fell beside it.
  drop(id, pos, yaw, retry = false, z = null) {
    const b = this.find(id);
    if (!b || this.holdingId !== id) return false;
    this.holdingId = null;
    b.pos = pos.slice(0, 2);
    b.yaw = yaw;
    this.settle(b, z);
    this.pickId = retry ? id : null;
    this.regrasp = retry ? id : null;
    if (!retry) this.placeTarget = null;
    this.walkTo = null;
    return true;
  }
  // A box the physics has at bottom height z: on the table under it when it is not far below the table's top.
  settle(b, z = null) {
    const s = this.support(b.pos);
    if (z !== null && s.table && z < s.z / 2) { b.z = 0; b.table = null; }
    else { b.z = s.z; b.table = s.table; }
  }
}

export function stickFromKeys(keys, yaw) {
  let f = 0, l = 0;
  if (keys.has('KeyW') || keys.has('ArrowUp')) f += 1;
  if (keys.has('KeyS') || keys.has('ArrowDown')) f -= 1;
  if (keys.has('KeyA') || keys.has('ArrowLeft')) l += 1;
  if (keys.has('KeyD') || keys.has('ArrowRight')) l -= 1;
  if (!f && !l) return null;
  const n = Math.hypot(f, l);
  f /= n; l /= n;
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return [c * f - s * l, s * f + c * l];
}

export class Sim {
  constructor(controller, footLock, cfg) {
    this.controller = controller;
    this.footLock = footLock;
    this.cfg = cfg;
    this.world = new World();
    this.scenario = null;
    this.simTime = 0;
    this.stats = { lowTime: 0, uprightTime: 0 };
    this.forceSearch = false;
    this.grip = null;
    this.parked = null;
    this.heldPose = null;
    this.renderJoints = null;
    this.onStep = null;
    this.speculative = false;
    // A box that leaves the hands while the robot rises with it stays the box to pick up (drop).
    this.retryLift = true;
    // The pick motion edits (pickmotion.js), or null to play the data's. supports: the height of what the box to pick
    // up and the place target stand on, kept after a grasp or a release while the robot rises from it.
    this.pickMotion = null;
    this.supports = { pick: 0, place: 0 };
  }

  // The replan modes of the physical mode (phys/runner.js) run the controller ahead and then back to a saved
  // state. The databases and settings are shared, not copied.
  saveState() {
    const c = this.controller, b = c.backend;
    const keep = new Set([this.cfg, c.cfg, c.p, b.cfg, b.ctrl, b.db, b.pose, b.L, b.steps, b.starts, b.walkRangesOf, b.edits, this.footLock.legs].filter(Boolean));
    return copyState(Object.fromEntries(SIM_STATE.map((k) => [k, this[k]])), keep);
  }

  // What the physical mode's pick and place check reads from a reference frame: the box being picked up or the
  // place of the held box, and whether the controller bends down for the grasp or the release. Also the state and
  // the cost of the playing clip (Controller.playCost) when the frame was made, for the evaluation.
  checkTag() {
    const w = this.world;
    const c = this.controller;
    return { target: w.holdingId !== null ? `place ${w.holdingId}` : w.pickId !== null ? `pick ${w.pickId}` : null, bending: c.bentForEvent(),
      state: c.fsm.state, cost: c.playCost() };
  }

  // Takes over the saved state, which must not be restored twice.
  restoreState(s) {
    for (const k of SIM_STATE) this[k] = s[k];
  }

  resetRobot() {
    this.controller.reset([0, 0, 0], 0, false);
    this.footLock.reset();
    this.grip = null;
    this.parked = null;
    this.heldPose = null;
    this.renderJoints = null;
  }

  startScripted() {
    this.world.reset(INITIAL_BOXES, INITIAL_TABLES);
    this.supports = { pick: 0, place: 0 };
    this.resetRobot();
    this.scenario = { index: 0, stepTime: 0, drive: null, driveDone: false, done: false, failed: null };
    this.simTime = 0;
    this.stats = { lowTime: 0, uprightTime: 0 };
  }

  startPlay(force = false) {
    if (force || this.world.boxes.length === 0 || this.scenario) {
      this.world.reset(PLAY_BOXES, PLAY_TABLES);
      this.supports = { pick: 0, place: 0 };
      this.resetRobot();
    }
    this.scenario = null;
  }

  get done() { return !!(this.scenario && this.scenario.done); }
  // { reason: 'drop', box, state, t } once the scripted sequence ended as a failure (drop).
  get failed() { return (this.scenario && this.scenario.failed) || null; }

  // A point relative to the robot: `ahead` metres along its facing direction, `left` to its left.
  relative(ahead, left) {
    const P = this.controller.pose, yaw = q.yaw(P.rootRot);
    return { pos: [P.rootPos[0] + ahead * Math.cos(yaw) - left * Math.sin(yaw), P.rootPos[1] + ahead * Math.sin(yaw) + left * Math.cos(yaw)], yaw };
  }

  scenarioApi() {
    const sim = this, w = this.world;
    return {
      setPick: (k) => w.setPick(k),
      setPlaceAhead: (ahead, left, turn = 0) => { const r = sim.relative(ahead, left); w.setPlace(r.pos, r.yaw + turn); sim.forceSearch = true; },
      movePlaceBy: (dx, dy, turn = 0) => {
        const t = w.placeTarget;
        if (t) { w.setPlace([t.pos[0] + dx, t.pos[1] + dy], t.yaw + turn); sim.forceSearch = true; }
      },
      setPlaceNear: (k, dx, dy, yaw) => { const b = w.find(k); if (b) { w.setPlace([b.pos[0] + dx, b.pos[1] + dy], yaw); sim.forceSearch = true; } },
      moveBox: (k, p, y) => { w.moveBox(k, p, y); sim.forceSearch = true; },
      moveBoxBy: (k, dx, dy, turn = 0) => { const b = w.find(k); if (b) { w.moveBox(k, [b.pos[0] + dx, b.pos[1] + dy], b.yaw + turn); sim.forceSearch = true; } },
      setPlaceOnTable: (t, edge, along) => { if (w.setPlaceOnTable(t, edge, along)) sim.forceSearch = true; },
      removeBox: (k) => { w.removeBox(k); sim.forceSearch = true; },
      addBox: (k, p, y) => w.addBox(k, p, y),
      drive: (segments) => {
        w.clearTargets();
        sim.scenario.drive = { segments, i: 0, t: 0, yaw: q.yaw(sim.controller.pose.rootRot) };
        sim.scenario.driveDone = false;
      },
      finish: () => { sim.scenario.done = true; },
    };
  }

  // While the robot picks a box up again after a failed lift (physical mode), the sequence waits. A failed sequence
  // makes no more steps.
  tickScenario(dt) {
    const sc = this.scenario;
    if (!sc || sc.done || sc.failed) return;
    const waiting = this.world.regrasp !== null;
    if (!waiting) sc.stepTime += dt;
    if (sc.drive) {
      const d = sc.drive;
      d.t += dt;
      if (d.t > d.segments[d.i][0]) { d.i++; d.t = 0; }
      if (d.i >= d.segments.length) { sc.drive = null; sc.driveDone = true; sc.stepTime = 0; }
    }
    if (waiting) return;
    const step = SCENARIO[sc.index];
    if (!step) return;
    const c = this.controller;
    const ctx = { ...this.stats, stepTime: sc.stepTime, mode: c.mode, phase: c.boxPhase(), driveDone: sc.driveDone,
                  holding: this.world.holdingId !== null, hipz: c.hipHeight() };
    if (step.when(ctx)) {
      step.act(this.scenarioApi());
      sc.index++;
      sc.stepTime = 0;
      if (this.onStep && !this.speculative) this.onStep(step, sc.index);
    }
  }

  floorPose(box) {
    return { pos: [box.pos[0], box.pos[1], (box.z ?? 0) + BOX.size[2] / 2], rot: q.fromYaw(box.yaw) };
  }

  // The tables as Controller.update takes them.
  tableObstacles() {
    return this.world.tables.map((t) => ({ id: t.id, pos: t.pos, yaw: t.yaw, half: [t.size[0] / 2, t.size[1] / 2], table: true }));
  }

  // The floor boxes and the tables as Locomotion steers around them (Controller.collide).
  obstacles() {
    const half = [BOX.size[0] / 2, BOX.size[1] / 2];
    return this.world.groundBoxes().map((b) => ({ id: b.id, pos: b.pos, yaw: b.yaw, half })).concat(this.tableObstacles());
  }

  // The pick and place motions for what the box to pick up and the place target stand on (pickmotion.js). While a bend
  // plays, both clips get its height and its manual edit (Controller.bendOf). A change of the clip that plays is blended
  // in as a transition.
  syncPickMotion() {
    const w = this.world, pb = w.pickBox(), pm = this.pickMotion, c = this.controller;
    if (pb) this.supports.pick = pb.z ?? 0;
    if (w.placeTarget) this.supports.place = w.placeTarget.z ?? 0;
    if (!pm) return;
    const bend = c.bendOf(), s = this.supports;
    const changed = pm.sync(s, BOX.size, bend);
    c.boxEdit = pm.offsets();
    const b = c.backend;
    if (!changed.length || !b.active || b.active === 'loco') return;
    const clip = pm.clipOf(b.db[b.active].idx[b.frame]);
    if (clip && changed.includes(clip.kind)) c.transition(b.matchPose({ name: b.active, frame: b.frame }, c.curr.rootPos, c.curr.rootRot, c.dt));
  }

  // stick and face: the player's stick and facing ([x, y] world directions or null), used outside the scripted sequence.
  step(stick = null, slow = false, face = null, walkTo = null) {
    const dt = this.controller.dt, w = this.world;
    if (this.scenario) this.tickScenario(dt);
    this.syncPickMotion();
    const drive = this.scenario && this.scenario.drive;
    const pickBox = w.pickBox(), n = this.cfg.box_sides;
    if (pickBox) pickBox.sides = w.sidesFor(pickBox, n);
    if (w.placeTarget) w.placeTarget.sides = w.sidesFor(w.placeTarget, n);
    const input = {
      stick: drive ? this.driveStick() : stick,
      face: drive ? null : face,
      walkTo: drive ? null : walkTo,
      slow,
      holding: w.holdingId !== null, pickBox, placeTarget: w.placeTarget,
      boxes: w.groundBoxes(), boxHalf: [BOX.size[0] / 2, BOX.size[1] / 2], tables: this.tableObstacles(),
      forceSearch: this.forceSearch,
    };
    this.forceSearch = false;
    const c = this.controller;
    c.update(input);

    if (c.events.grasp && input.pickBox) {
      // The box looks the same turned by half a turn (a quarter turn with 4 sides, where it is square), and the
      // held box X axis points away from the carrier, so the blend starts from the copy of the floor pose closest to
      // the hands.
      const box = input.pickBox, turn = 2 * Math.PI / this.cfg.box_sides;
      const x = q.mulVec(c.heldBoxWorld().rot, [1, 0, 0]);
      box.yaw = wrapAngle(box.yaw + turn * Math.round(wrapAngle(Math.atan2(x[1], x[0]) - box.yaw) / turn));
      w.holdingId = box.id;
      w.pickId = null;
      if (w.regrasp === box.id) w.regrasp = null;
      this.grip = { id: box.id, from: this.floorPose(box), t: 0, landing: false };
    }
    if (c.events.release && w.holdingId !== null) {
      // The box comes to rest on the place target, as in the data, where the rest pose after the
      // release is the target.
      const box = w.find(w.holdingId);
      const hb = c.heldBoxWorld();
      const t = c.commit || c.target;
      const x = q.mulVec(hb.rot, [1, 0, 0]);
      box.pos = t ? t.pos.slice() : [hb.pos[0], hb.pos[1]];
      box.yaw = t ? t.yaw : Math.atan2(x[1], x[0]);
      if (w.placeTarget && t) { box.z = w.placeTarget.z ?? 0; box.table = w.placeTarget.table ?? null; } else w.rest(box);
      this.grip = { id: box.id, from: hb, t: 0, landing: true };
      w.holdingId = null;
      w.placeTarget = null;
    }
    this.heldPose = this.aboveFloor(this.updateHeld(dt));

    this.renderJoints = this.footLock.apply(c.renderState(), dt);
    this.stats.lowTime = c.isLow() ? this.stats.lowTime + dt : 0;
    this.stats.uprightTime = c.isUpright() ? this.stats.uprightTime + dt : 0;
    this.simTime += dt;
  }

  driveStick() {
    const d = this.scenario.drive, seg = d.segments[d.i][1];
    return stickFromKeys(new Set(seg.split('').map((k) => `Key${k}`)), d.yaw);
  }

  // The physical box left the hands: the hold ends where the physics has the box (physical mode). While the controller
  // rises with the box (Lifting the box) and retryLift is on, the box stays the box to pick up, so the robot picks it
  // up again. Any other drop leaves no target, and the scripted sequence ends as a failure.
  // Returns false, or { state: the controller's state at the drop, retry }.
  drop(id, pos, yaw, z = null) {
    const c = this.controller, state = c.fsm.state, box = this.world.find(id);
    const retry = this.retryLift && c.boxPhase() === 'pick-post';
    if (!this.world.drop(id, pos, yaw, retry, z)) return false;
    this.grip = null;
    this.parked = null;
    this.heldPose = null;
    this.forceSearch = true;
    const sc = this.scenario;
    if (!retry && sc && !sc.done && !sc.failed) {
      sc.failed = { reason: 'drop', box: box.label, state, t: this.simTime };
      sc.drive = null;
    }
    return { state, retry };
  }

  // At a release the box pose of the playing frames can dip below the floor for a few frames. The drawn box is
  // lifted until its lowest corner is on the floor, or on the table under its centre.
  aboveFloor(pose) {
    if (!pose) return pose;
    const half = BOX.size.map((s) => s / 2);
    const depth = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].reduce((s, e, i) => s + Math.abs(q.mulVec(pose.rot, e)[2]) * half[i], 0);
    const lift = this.world.support(pose.pos).z + depth - pose.pos[2];
    return lift > 0 ? { ...pose, pos: [pose.pos[0], pose.pos[1], pose.pos[2] + lift] } : pose;
  }

  flatOnFloor(pose) {
    const x = q.mulVec(pose.rot, [1, 0, 0]);
    return { pos: [pose.pos[0], pose.pos[1], this.world.support(pose.pos).z + BOX.size[2] / 2], rot: q.fromYaw(Math.atan2(x[1], x[0])) };
  }

  // World pose of the box in the robot's hands, blended at the grasp and settled at the release. When
  // an aborted place briefly sets the box down in the animation (held output low while the robot still
  // holds the box), the box stays parked on the floor until the hands take it again.
  updateHeld(dt) {
    const w = this.world, c = this.controller;
    let g = this.grip;
    if (g) { g.t += dt; if (g.t > 0.3) { this.grip = g = null; } }
    if (g && g.landing) {
      const box = w.find(g.id);
      if (!box) return null;
      const to = this.floorPose(box), k = smoothstep(g.t / 0.25);
      return { id: g.id, pos: v3.lerp(g.from.pos, to.pos, k), rot: q.slerpShortestApprox(g.from.rot, to.rot, k) };
    }
    if (w.holdingId === null) { this.parked = null; return null; }
    const anim = c.heldBoxWorld();
    if (c.pose.held < 0.5) {
      if (!this.parked) {
        const last = this.heldPose && this.heldPose.id === w.holdingId ? this.heldPose : anim;
        this.parked = this.flatOnFloor(last);
      }
      return { id: w.holdingId, ...this.parked };
    }
    if (this.parked) {
      this.grip = g = { id: w.holdingId, from: this.parked, t: 0, landing: false };
      this.parked = null;
    }
    if (g && g.id === w.holdingId) {
      const k = smoothstep(g.t / 0.3);
      return { id: g.id, pos: v3.lerp(g.from.pos, anim.pos, k), rot: q.slerpShortestApprox(g.from.rot, anim.rot, k) };
    }
    return { id: w.holdingId, ...anim };
  }
}
