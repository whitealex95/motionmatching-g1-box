// The kinematic side without the page: the boxes, the scripted sequence and the controller step with the
// grasp and the release. The page (app.js) and the headless evaluation (tools/eval_physical.mjs) both drive it.

import { SCENARIO, INITIAL_BOXES } from './scenario.js';
import { q, v3, wrapAngle } from './math.js';

// The box of the holosoma retargets: X depth away from the carrier, Y between the palms, Z height.
// The page sets it from the data variant's config.
export const BOX = { size: [0.30, 0.38, 0.30] };

export const PLAY_BOXES = [
  { label: 'A', pos: [1.8, 0.5], yaw: 0.2 },
  { label: 'B', pos: [-1.6, 1.8], yaw: 0.9 },
  { label: 'C', pos: [0.8, -2.0], yaw: -0.4 },
];

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

const SIM_STATE = ['controller', 'footLock', 'world', 'scenario', 'simTime', 'stats', 'forceSearch', 'grip', 'parked', 'heldPose', 'renderJoints'];

// Boxes carry a revision that counts the moves made by commands (the scripted sequence, the player). The
// physical mode places a box in the physics only when its revision changes, so the moves that follow the
// physics (the grasp, the release and the boxes fed back from the physics) never teleport a box.
export class World {
  constructor() { this.reset([]); }
  reset(boxes) {
    this.boxes = [];
    this.nextId = 1;
    this.pickId = null;
    this.holdingId = null;
    this.placeTarget = null;
    this.selectedId = null;
    this.walkTo = null;
    this.regrasp = null;
    for (const b of boxes) this.addBox(b.label, b.pos, b.yaw);
  }
  find(key) { return this.boxes.find((b) => b.id === key || b.label === key) || null; }
  floorBoxes() { return this.boxes.filter((b) => b.id !== this.holdingId); }
  addBox(label, pos, yaw = 0) {
    const used = new Set(this.boxes.map((b) => b.label));
    if (!label || used.has(label)) {
      label = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').find((c) => !used.has(c)) || String(this.nextId);
    }
    const b = { id: this.nextId++, label, pos: pos.slice(0, 2), yaw, rev: 0 };
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
    b.rev++;
  }
  rotateBox(key, d) { const b = this.find(key); if (b && b.id !== this.holdingId) { b.yaw += d; b.rev++; } }
  setPick(key) {
    const b = key === null ? null : this.find(key);
    if (this.holdingId !== null) return false;
    this.pickId = b ? b.id : null;
    this.walkTo = null;
    return true;
  }
  setPlace(pos, yaw) {
    if (this.holdingId === null) return false;
    this.placeTarget = { pos: pos.slice(0, 2), yaw };
    this.walkTo = null;
    return true;
  }
  clearTargets() { this.pickId = null; this.placeTarget = null; this.regrasp = null; }
  pickBox() { return this.pickId === null ? null : this.find(this.pickId); }
  // The physical box left the hands (physical mode): the box is where the physics has it, and the robot
  // picks it up again. A place target stays for after the new grasp.
  drop(id, pos, yaw) {
    const b = this.find(id);
    if (!b || this.holdingId !== id) return false;
    this.holdingId = null;
    b.pos = pos.slice(0, 2);
    b.yaw = yaw;
    this.pickId = id;
    this.regrasp = id;
    this.walkTo = null;
    return true;
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
  }

  // The replan modes of the physical mode (phys/runner.js) run the controller ahead and then back to a saved
  // state. The networks, databases and settings are shared, not copied.
  saveState() {
    const c = this.controller, b = c.backend;
    const keep = new Set([this.cfg, c.cfg, c.p, b.cfg, b.ctrl, b.nets, b.db, b.pose, b.L, b.steps, this.footLock.legs].filter(Boolean));
    return copyState(Object.fromEntries(SIM_STATE.map((k) => [k, this[k]])), keep);
  }

  // What the physical mode's pick and place check reads from a reference frame: the box being picked up or the
  // place of the held box, and whether the controller bends down for the grasp or the release.
  checkTag() {
    const w = this.world;
    return { target: w.holdingId !== null ? `place ${w.holdingId}` : w.pickId !== null ? `pick ${w.pickId}` : null, bending: this.controller.bentForEvent() };
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
    this.world.reset(INITIAL_BOXES);
    this.resetRobot();
    this.scenario = { index: 0, stepTime: 0, drive: null, driveDone: false, done: false };
    this.simTime = 0;
    this.stats = { lowTime: 0, uprightTime: 0 };
  }

  startPlay(force = false) {
    if (force || this.world.boxes.length === 0 || this.scenario) {
      this.world.reset(PLAY_BOXES);
      this.resetRobot();
    }
    this.scenario = null;
  }

  get done() { return !!(this.scenario && this.scenario.done); }

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

  // While the robot picks a dropped box up again (physical mode), the sequence waits.
  tickScenario(dt) {
    const sc = this.scenario;
    if (!sc || sc.done) return;
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
    return { pos: [box.pos[0], box.pos[1], BOX.size[2] / 2], rot: q.fromYaw(box.yaw) };
  }

  // stick: the player's stick, used outside the scripted sequence.
  step(stick = null, slow = false) {
    const dt = this.controller.dt, w = this.world;
    if (this.scenario) this.tickScenario(dt);
    const input = {
      stick: this.scenario && this.scenario.drive ? this.driveStick() : stick,
      slow,
      holding: w.holdingId !== null, pickBox: w.pickBox(), placeTarget: w.placeTarget,
      boxes: w.floorBoxes(), boxHalf: [BOX.size[0] / 2, BOX.size[1] / 2],
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
      // release is the target. The Decompressor's box output is least accurate at this moment.
      const box = w.find(w.holdingId);
      const hb = c.heldBoxWorld();
      const t = c.commit || c.target;
      const x = q.mulVec(hb.rot, [1, 0, 0]);
      box.pos = t ? t.pos.slice() : [hb.pos[0], hb.pos[1]];
      box.yaw = t ? t.yaw : Math.atan2(x[1], x[0]);
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

  // The physical box left the hands: the hold ends where the physics has the box (physical mode).
  drop(id, pos, yaw) {
    if (!this.world.drop(id, pos, yaw)) return false;
    this.grip = null;
    this.parked = null;
    this.heldPose = null;
    this.forceSearch = true;
    return true;
  }

  // At a release the box Stepper can overshoot the hold state, and the Decompressor's box output then dips
  // below the floor for a few frames. The drawn box is lifted until its lowest corner is on the floor.
  aboveFloor(pose) {
    if (!pose) return pose;
    const half = BOX.size.map((s) => s / 2);
    const depth = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].reduce((s, e, i) => s + Math.abs(q.mulVec(pose.rot, e)[2]) * half[i], 0);
    const lift = depth - pose.pos[2];
    return lift > 0 ? { ...pose, pos: [pose.pos[0], pose.pos[1], pose.pos[2] + lift] } : pose;
  }

  flatOnFloor(pose) {
    const x = q.mulVec(pose.rot, [1, 0, 0]);
    return { pos: [pose.pos[0], pose.pos[1], BOX.size[2] / 2], rot: q.fromYaw(Math.atan2(x[1], x[0])) };
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
