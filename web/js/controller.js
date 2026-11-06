// Learned Motion Matching controller for the G1 picking up, carrying and putting down a box, built like
// the chair behaviour of Section 5 of the paper. A locomotion controller (27 features plus an
// empty-handed or carrying tag) runs by default. Every search the system also tries to match the box
// set (relative box position and direction, a pick or place tag and the hold state). A box match below a cost
// threshold switches to the box controller, which keeps re-searching. Root warping makes the hands
// arrive exactly at the box. Basic Motion Matching (exact search) is the reference.
//
// The box controller has four states, set by its tag and by whether the robot holds the box:
//   pick, empty      walk in and bend down to the box. The Decompressor's held output rising is the grasp.
//   pick, holding    rise with the box.
//   place, holding   walk in and lower the box. The held output falling is the release.
//   place, empty     rise without the box.
// If the box (or the place target) moves while the robot is bent, the robot switches to the other tag's
// rising state: it stands up empty-handed (pick aborted) or still holding (place aborted).
//
// Ports controller.cpp / lmm.h / database.h of orangeduck/Motion-Matching, with Z-up and +X forward.

import { v3, q, dampFactor, simpleSpringDamperQuat, decaySpringVec, decaySpringQuat, decaySpringScalar, wrapAngle } from './math.js';

export const NDOF = 29;
const NPOSE_FEATURES = 15;

export const DEFAULTS = {
  searchTime: 0.1,
  inertializeHalflife: 0.1,
  simVelocityHalflife: 0.27,
  simRotationHalflife: 0.27,
  walkFwd: 0.8, walkSide: 0.45, walkBack: 0.4,
  slowScale: 0.5,
  adjustPosHalflife: 0.1, adjustRotHalflife: 0.2, adjustPosMaxRatio: 0.5, adjustRotMaxRatio: 0.5,
  clampMaxDistance: 0.15, clampMaxAngle: 0.5 * Math.PI,
  boxEnter: { lmm: 2.5, mm: 2.5 },
  boxLeave: { lmm: 4.0, mm: 4.0 },
  forceEngageRadius: 1.1,
  sideMargin: { lmm: 0.25, mm: 0.25 },
  // Exact search only. 'different' searches the pick and the place frames as two Matching Databases.
  boxDb: 'same',
  // Exact search only. 'emm' searches the EMM locomotion takes instead of the empty-handed NSM frames, 'emmcarry' the
  // carry spans of EMM's box take instead of the carrying frames, and 'emmboth' both (setLoco).
  loco: 'nsm',
  engageRadius: 3.0,
  boxQueryRadius: 3.0,
  warpHalflife: 0.2, warpMaxRatio: 0.8, warpMaxDist: 0.7, warpMaxAngle: 1.2,
  // Exact search only: the box clips (walk in, bend, grasp or release, rise) play at this share of their speed.
  pickPlaceSpeed: 1,
  // 'clip': all frames of the box clips. 'step': only those from the last step before the grasp or the release to
  // the first step after it (lastSteps).
  pickPlaceSlow: 'clip',
  // 'always': root warping moves the root toward the target through the whole walk in and bend. 'step': not from the
  // last step before the grasp or the release on (exact search only), where both feet of the clip stay down.
  // 'pickstep': as 'step' before a grasp, as 'always' before a release.
  // config.json's warp sets it for a variant.
  warp: 'always',
  // Upright before the bend, a target this far (m) from where it was when the box controller took over sends the robot
  // back to Locomotion, which may not take the box controller again within 1.1 m for reapproachTime. Until the clip
  // that takes it again reaches its last step, the search jumps only to frames before a last step (walkInMask).
  // null: never.
  // config.json's leave_moved_m sets it for a variant.
  leaveMoved: null, reapproachTime: 1.0,
  // Exact search only. 'walk': a box range that ends before its clip's grasp or release (the walk of an ApproachPick v4
  // clip) is played on from entryLead before its last step (lastSteps) but never searched there. With config.json's
  // scenebot_entry_frames, a range of a SceneBot clip that holds its grasp or release is searched only in its first
  // scenebot_entry_frames frames and from the event on, and while a SceneBot pick or place plays before its grasp or
  // release the search jumps to no frame before a grasp or a release. 'any': every frame.
  boxEntry: 'walk', entryLead: 30,
  // Exact search only. 'search': the box controller searches every searchTime while it rises. 'finish': once the
  // playing frame is at or past its clip's grasp (a pick range) or release (a place range), the robot rises without a
  // search until it is up (risen) or the range ends. A clip that went past the event without it (the box out of
  // reach, or the physical grip lost) then sends the robot back to Locomotion to walk in again, as leaveMoved does.
  // config.json's rise sets it for a variant.
  rise: 'search',
  // Exact search only. transition_cost of motion_matching_search in Holden's Motion-Matching (database.h): added to the
  // squared cost of every frame but the playing one, so a jump must beat staying on the clip by this much.
  // config.json's transition_cost sets it for a variant.
  transitionCost: 0,
  zLow: 0.66, zUpright: 0.72,
  // The rise after a grasp or a release ends once the pelvis is above zUpright and, unless null, the torso is within
  // this tilt of vertical (deg). config.json's upright_tilt_deg sets it for a variant.
  uprightTilt: null,
  graspOn: 0.6, releaseOff: 0.4, graspReach: 0.25, floorBoxLift: 0.15,
  boxTransitionCost: 1.0, holdSwitchMax: 0.5,
  movedTolerance: 0.12, movedAngle: 0.35,
  collisionRadius: 0.36,
  standOff: 0.36, approachClearance: 0.3, approachFallback: 0.6,
};

// The bodies from the pelvis to torso_link.
export function torsoChain(bodies) {
  const chain = [];
  for (let k = bodies.findIndex((x) => x.name === 'torso_link'); k >= 0; k = bodies[k].parent) chain.unshift(bodies[k]);
  return chain;
}

// Torso link tilt from vertical (deg), from the pelvis rotation and angle(dof) of the joints.
export function torsoTilt(chain, hipRot, angle) {
  let r = q.norm(hipRot);
  for (const body of chain) {
    if (body.parent < 0) continue;
    r = q.mul(r, body.quat);
    if (body.axis) r = q.mul(r, q.fromAngleAxis(angle(body.dof), body.axis));
  }
  return Math.acos(Math.max(-1, Math.min(1, q.mulVec(r, [0, 0, 1])[2]))) * 180 / Math.PI;
}

// The three queries and the Matching Database each one searches.
export const QUERY_DB = { loco: 'loco', pick: 'box', place: 'box' };

function makePose() {
  return {
    rootPos: [0, 0, 0], rootRot: q.id(), rootVel: [0, 0, 0], rootAng: [0, 0, 0],
    hipPos: [0, 0, 0.77], hipRot: q.id(), hipVel: [0, 0, 0], hipAng: [0, 0, 0],
    joints: new Float64Array(NDOF), jointVel: new Float64Array(NDOF),
    contacts: [true, true], held: 0, boxPos: [0.3, 0, 0.7], boxRot: q.id(),
  };
}

function copyPose(p) {
  return {
    rootPos: p.rootPos.slice(), rootRot: p.rootRot.slice(), rootVel: p.rootVel.slice(), rootAng: p.rootAng.slice(),
    hipPos: p.hipPos.slice(), hipRot: p.hipRot.slice(), hipVel: p.hipVel.slice(), hipAng: p.hipAng.slice(),
    joints: Float64Array.from(p.joints), jointVel: Float64Array.from(p.jointVel),
    contacts: p.contacts.slice(), held: p.held, boxPos: p.boxPos.slice(), boxRot: p.boxRot.slice(),
  };
}

// The n sides a box is picked up from (2 or 4, a setting of the page): the box X axis points away from the
// carrier, and side k turns it by k / n of a full turn.
export function targetSides(pos, yaw, n) {
  return Array.from({ length: n }, (_, side) => {
    const y = yaw + side * 2 * Math.PI / n;
    return { side, yaw: y, dir: [Math.cos(y), Math.sin(y)], pos: pos.slice(0, 2) };
  });
}

class Backend {
  // Shared feature bookkeeping for both the learned and the exact backends.
  constructor(cfg) {
    this.cfg = cfg;
    this.ctrl = {};
    for (const name of ['loco', 'box']) {
      const c = cfg.controllers[name];
      this.ctrl[name] = { ...c, offset: Float32Array.from(c.offset), scale: Float32Array.from(c.scale) };
    }
  }
  normalize(name, query) {
    const c = this.ctrl[name], out = new Float32Array(c.nfeatures);
    for (let i = 0; i < c.nfeatures; i++) out[i] = (query[i] - c.offset[i]) / c.scale[i];
    return out;
  }
  denormalize(name, x) {
    const c = this.ctrl[name], out = new Float32Array(c.nfeatures);
    for (let i = 0; i < c.nfeatures; i++) out[i] = x[i] * c.scale[i] + c.offset[i];
    return out;
  }
}

export class LMMBackend extends Backend {
  constructor(cfg, nets) {
    super(cfg);
    this.kind = 'lmm';
    this.nets = nets;
    this.active = 'loco';
    this.features = new Float32Array(cfg.controllers.loco.nfeatures);
    this.latent = new Float32Array(cfg.nlatent);
  }
  currentFeatures() { return this.features; }
  // query is normalized; returns the projected features, latent and the matching cost.
  match(name, qn) {
    const n = this.ctrl[name].nfeatures;
    const out = this.nets[name].projector.evaluate(qn);
    const features = Float32Array.from(out.subarray(0, n));
    const latent = Float32Array.from(out.subarray(n, n + this.cfg.nlatent));
    let cost = 0;
    for (let i = 0; i < n; i++) cost += (qn[i] - features[i]) ** 2;
    return { name, features, latent, cost: Math.sqrt(cost) };
  }
  commit(m) {
    this.active = m.name;
    this.features = m.features;
    this.latent = m.latent;
  }
  matchPose(m, rootPos, rootRot, dt) {
    return this.decode(m.name, m.features, m.latent, rootPos, rootRot, dt);
  }
  step(dt) {
    const net = this.nets[this.active].stepper, n = this.features.length;
    const inp = new Float32Array(n + this.latent.length);
    inp.set(this.features); inp.set(this.latent, n);
    const out = net.evaluate(inp);
    for (let i = 0; i < n; i++) this.features[i] += dt * out[i];
    for (let i = 0; i < this.latent.length; i++) this.latent[i] += dt * out[n + i];
  }
  currentPose(rootPos, rootRot, dt) {
    return this.decode(this.active, this.features, this.latent, rootPos, rootRot, dt);
  }
  decode(name, features, latent, rootPos, rootRot, dt) {
    const inp = new Float32Array(features.length + latent.length);
    inp.set(features); inp.set(latent, features.length);
    const o = this.nets[name].decompressor.evaluate(inp);
    const p = makePose();
    p.hipPos = [o[0], o[1], o[2]];
    p.hipRot = q.fromXformXY([o[3], o[4], o[5], o[6], o[7], o[8]]);
    p.hipVel = [o[9], o[10], o[11]];
    p.hipAng = [o[12], o[13], o[14]];
    for (let i = 0; i < NDOF; i++) { p.joints[i] = o[15 + i]; p.jointVel[i] = o[44 + i]; }
    p.rootVel = q.mulVec(rootRot, [o[73], o[74], 0]);
    p.rootAng = q.mulVec(rootRot, [0, 0, o[78]]);
    p.rootPos = v3.add(rootPos, v3.scale(p.rootVel, dt));
    p.rootRot = q.norm(q.mul(q.fromScaledAngleAxis(v3.scale(p.rootAng, dt)), rootRot));
    p.contacts = [o[79] > 0.5, o[80] > 0.5];
    p.held = o[81];
    p.boxPos = [o[82], o[83], o[84]];
    p.boxRot = q.fromXformXY([o[85], o[86], o[87], o[88], o[89], o[90]]);
    return p;
  }
}

export class MMBackend extends Backend {
  constructor(cfg, data) {
    super(cfg);
    this.kind = 'mm';
    this.pose = data.pose;  // Float32Array (nframes * width)
    this.width = cfg.pose_width;
    this.L = cfg.pose_layout;
    this.db = data.db;      // per controller: X, idx, ranges, rangeStop
    for (const name in data.db) {
      if (!this.ctrl[name]) this.ctrl[name] = { ...this.ctrl.box, offset: data.db[name].offset, scale: data.db[name].scale };
    }
    // The Locomotion Matching Databases, 'nsm' (the variant's) and 'emm' once loadEmmLoco has run.
    this.locos = { nsm: { ...data.db.loco, offset: this.ctrl.loco.offset, scale: this.ctrl.loco.scale } };
    this.loco = 'nsm';
    this.active = 'loco';
    this.frame = 0;         // index into the active controller's matching database
    this.frac = 0;          // how far the playhead is from frame toward frame + 1, at a speed below 1
    this.steps = lastSteps(cfg, data.pose);
  }
  // A frame of a box Matching Database between the last step before its grasp or release and the first step after.
  planted(name, frame) {
    return name !== 'loco' && this.steps.planted[this.db[name].idx[frame]] === 1;
  }
  // The frame index of the old database means nothing in the new one, and past its end it gave NaN poses.
  useLoco(name) {
    const l = this.locos[name];
    this.db.loco = l;
    this.ctrl.loco = { ...this.ctrl.loco, offset: l.offset, scale: l.scale };
    this.loco = name;
    if (this.active === 'loco') this.active = null;
  }
  currentFeatures() {
    const d = this.db[this.active], n = d.nfeatures, a = d.X.subarray(this.frame * n, (this.frame + 1) * n);
    if (!this.frac) return a;
    const b = d.X.subarray((this.frame + 1) * n, (this.frame + 2) * n);
    return a.map((x, i) => x + this.frac * (b[i] - x));
  }
  endOfAnim() {
    const d = this.db[this.active];
    return this.frame + 1 >= d.rangeStop[this.frame];
  }
  // The clip of box range r: its name, frames and grasp (a pick range) or release (a place range).
  rangeClip(name, r) {
    const d = this.db[name], rows = this.cfg.clips.rows, g0 = d.idx[d.ranges[2 * r]];
    let lo = 0, hi = rows.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (rows[mid][2] <= g0) lo = mid; else hi = mid - 1;
    }
    const [n, , start, stop, attach, release] = rows[lo];
    return { name: this.cfg.clips.names[n], start, stop, event: (d.rangeKind ? d.rangeKind[r] : name === 'place') ? release : attach };
  }
  // Per frame of a box Matching Database: 1 in a range that ends before its clip's grasp (pick) or release (place),
  // from entryLead frames before the clip's last step on, and in the middle of a SceneBot clip (scenebot_entry_frames).
  // The EMM frames after the variant's clips are never marked.
  noEntry(name) {
    this.noEntryOf ??= {};
    if (this.noEntryOf[name]) return this.noEntryOf[name];
    const d = this.db[name], out = new Uint8Array(d.idx.length), sbn = this.cfg.scenebot_entry_frames;
    for (let r = 0; r < d.ranges.length / 2; r++) {
      const a = d.ranges[2 * r], b = d.ranges[2 * r + 1], g0 = d.idx[a], g1 = d.idx[b - 1];
      const { name: clip, start, stop, event } = this.rangeClip(name, r);
      if (g0 >= stop || !(event >= 0)) continue;
      if (g1 - start >= event) {
        if (sbn && clip.startsWith('SceneBot')) for (let i = a + sbn; i < b && d.idx[i] - start < event; i++) out[i] = 1;
        continue;
      }
      let first = a;
      while (first < b && !this.steps.planted[d.idx[first]]) first++;
      if (first < b) out.fill(1, Math.max(a, first - DEFAULTS.entryLead), b);
    }
    return (this.noEntryOf[name] = out);
  }
  // Per frame of a box Matching Database: 1 at or past the grasp (a pick range) or the release (a place range) of its clip.
  afterEvent(name) {
    this.afterEventOf ??= {};
    if (this.afterEventOf[name]) return this.afterEventOf[name];
    const d = this.db[name], out = new Uint8Array(d.idx.length);
    for (let r = 0; r < d.ranges.length / 2; r++) {
      const { start, event } = this.rangeClip(name, r);
      if (!(event >= 0)) continue;
      for (let i = d.ranges[2 * r]; i < d.ranges[2 * r + 1]; i++) if (d.idx[i] - start >= event) out[i] = 1;
    }
    return (this.afterEventOf[name] = out);
  }
  // Per frame of a box Matching Database: 1 in a range of a SceneBot pick or place before its grasp or release.
  sceneBotPre(name) {
    this.sceneBotPreOf ??= {};
    if (this.sceneBotPreOf[name]) return this.sceneBotPreOf[name];
    const d = this.db[name], out = new Uint8Array(d.idx.length);
    for (let r = 0; r < d.ranges.length / 2; r++) {
      const { name: clip, start, event } = this.rangeClip(name, r);
      if (!clip.startsWith('SceneBot') || !(event >= 0)) continue;
      for (let i = d.ranges[2 * r]; i < d.ranges[2 * r + 1]; i++) if (d.idx[i] - start < event) out[i] = 1;
    }
    return (this.sceneBotPreOf[name] = out);
  }
  // noEntry, and every frame before the grasp or the release of its clip.
  beforeEventMask(name) {
    this.beforeEventOf ??= {};
    if (this.beforeEventOf[name]) return this.beforeEventOf[name];
    const out = this.noEntry(name).slice(), ae = this.afterEvent(name);
    for (let i = 0; i < out.length; i++) if (!ae[i]) out[i] = 1;
    return (this.beforeEventOf[name] = out);
  }
  // noEntry, and for a kind (pick or place) that has walks, also every frame from a last step to the first step after.
  // Searched after the target moved and sent the robot back to Locomotion, so that it walks in again.
  walkInMask(name) {
    this.walkInOf ??= {};
    if (this.walkInOf[name]) return this.walkInOf[name];
    const d = this.db[name], ne = this.noEntry(name), out = ne.slice(), walks = [false, false];
    const kind = (r) => (d.rangeKind ? d.rangeKind[r] : name === 'place' ? 1 : 0);
    for (let r = 0; r < d.ranges.length / 2; r++) if (ne[d.ranges[2 * r + 1] - 1]) walks[kind(r)] = true;
    for (let r = 0; r < d.ranges.length / 2; r++) {
      if (!walks[kind(r)]) continue;
      for (let i = d.ranges[2 * r]; i < d.ranges[2 * r + 1]; i++) if (this.steps.planted[d.idx[i]]) out[i] = 1;
    }
    return (this.walkInOf[name] = out);
  }

  // entry > 0 searches only the first entry frames of the place ranges. mask: frames not to jump to.
  match(name, qn, gain, ignoreRangeEnd = 20, ignoreSurrounding = 20, entry = 0, mask = null, transitionCost = 0) {
    const d = this.db[name], n = d.nfeatures, X = d.X;
    const cur = name === this.active && !this.endOfAnim() ? this.frame : -1;
    let best = cur, bestCost = Infinity;
    if (cur >= 0) {
      // Between two frames (a pick and place speed below 1) the query's pose comes from the blended features.
      const f = this.currentFeatures();
      bestCost = 0;
      for (let j = 0; j < n; j++) bestCost += gain[j] * (qn[j] - f[j]) ** 2;
    }
    for (let r = 0; r < d.ranges.length / 2; r++) {
      if (entry && d.rangeKind && d.rangeKind[r] !== 1) continue;
      const a = d.ranges[2 * r], b = Math.min(d.ranges[2 * r + 1] - ignoreRangeEnd, entry ? a + entry : Infinity);
      for (let i = a; i < b; i++) {
        if ((cur >= 0 && Math.abs(i - cur) < ignoreSurrounding) || (mask && mask[i])) continue;
        let c = transitionCost;
        const o = i * n;
        for (let j = 0; j < n && c < bestCost; j++) { const e = qn[j] - X[o + j]; c += gain[j] * e * e; }
        if (c < bestCost) { bestCost = c; best = i; }
      }
    }
    const cost = Math.sqrt(best === cur ? bestCost : bestCost - transitionCost);
    return { name, frame: best, cost, features: X.subarray(best * n, (best + 1) * n), same: best === cur };
  }
  commit(m) {
    this.active = m.name;
    this.frame = m.frame;
    this.frac = 0;
  }
  row(name, frame) {
    const g = this.db[name].idx[frame];
    return this.pose.subarray(g * this.width, (g + 1) * this.width);
  }
  rowPose(r, rootPos, rootRot) {
    const L = this.L, p = makePose();
    const vec = (k, n) => Array.from(r.subarray(L[k][0], L[k][0] + n));
    p.hipPos = vec('hip_pos', 3);
    p.hipRot = q.norm(vec('hip_quat', 4));
    p.hipVel = vec('hip_vel', 3);
    p.hipAng = vec('hip_ang', 3);
    for (let i = 0; i < NDOF; i++) { p.joints[i] = r[L.joints[0] + i]; p.jointVel[i] = r[L.joint_vel[0] + i]; }
    p.rootVel = q.mulVec(rootRot, [r[L.root_vel[0]], r[L.root_vel[0] + 1], 0]);
    p.rootAng = q.mulVec(rootRot, [0, 0, r[L.root_ang[0] + 2]]);
    p.rootPos = rootPos.slice();
    p.rootRot = rootRot.slice();
    p.contacts = [r[L.contacts[0]] > 0.5, r[L.contacts[0] + 1] > 0.5];
    p.held = r[L.held[0]];
    p.boxPos = vec('box_pos', 3);
    p.boxRot = q.norm(vec('box_quat', 4));
    return p;
  }
  matchPose(m, rootPos, rootRot, dt, speed = 1) {
    return this.scaleVelocities(this.rowPose(this.row(m.name, m.frame), rootPos, rootRot), speed);
  }
  step() {}
  // speed < 1 moves the playhead that many frames per step: the root moves by that share of the frame's root delta,
  // the pose is blended between the two frames around the playhead, and its velocities are scaled by speed.
  currentPose(rootPos, rootRot, dt, speed = 1) {
    const L = this.L;
    let pos = rootPos, rot = rootRot, left = speed;
    while (left > 1e-9) {
      const prev = this.row(this.active, this.frame), s = Math.min(left, 1 - this.frac);
      pos = v3.add(pos, q.mulVec(rot, [s * prev[L.root_delta[0]], s * prev[L.root_delta[0] + 1], 0]));
      rot = q.norm(q.mul(q.fromYaw(s * prev[L.root_delta[0] + 2]), rot));
      left -= s;
      if (this.endOfAnim()) { this.frac = 0; break; }
      this.frac += s;
      if (this.frac > 1 - 1e-9) { this.frame++; this.frac = 0; }
    }
    const p = this.rowPose(this.row(this.active, this.frame), pos, rot);
    if (speed === 1 && !this.frac) return p;
    const t = this.frac;
    if (t) {
      const n = this.rowPose(this.row(this.active, this.frame + 1), pos, rot);
      const lerp = (a, b) => a.map((x, i) => x + t * (b[i] - x));
      p.hipPos = lerp(p.hipPos, n.hipPos);
      p.hipRot = q.slerpShortestApprox(p.hipRot, n.hipRot, t);
      p.boxPos = lerp(p.boxPos, n.boxPos);
      p.boxRot = q.slerpShortestApprox(p.boxRot, n.boxRot, t);
      p.held += t * (n.held - p.held);
      if (t >= 0.5) p.contacts = n.contacts;
      for (let i = 0; i < NDOF; i++) {
        p.joints[i] += t * (n.joints[i] - p.joints[i]);
        p.jointVel[i] += t * (n.jointVel[i] - p.jointVel[i]);
      }
      for (const k of ['hipVel', 'hipAng', 'rootVel', 'rootAng']) p[k] = lerp(p[k], n[k]);
    }
    return this.scaleVelocities(p, speed);
  }
  scaleVelocities(p, speed) {
    if (speed === 1) return p;
    for (let i = 0; i < NDOF; i++) p.jointVel[i] *= speed;
    for (const k of ['hipVel', 'hipAng', 'rootVel', 'rootAng']) p[k] = v3.scale(p[k], speed);
    return p;
  }
}

// The pose's contacts are toe speed labels (lmm/database.py). Both feet down with lifts shorter than MIN_LIFT frames
// filled is a stance. The last step before a grasp or a release is the first frame of the stance around it, or of the
// stance that starts within LATE_STEP frames after it when a foot is in the air at the event. The stance's end is
// the first step after it. Per clip row: [{ kind, event, step, lift }] in frames of the clip. planted marks the
// frames of the Animation Database from a last step to its first step after.
const MIN_LIFT = 8, LATE_STEP = 30;
export function lastSteps(cfg, pose) {
  const W = cfg.pose_width, C = cfg.pose_layout.contacts[0];
  const planted = new Uint8Array(cfg.nframes);
  const clips = cfg.clips.rows.map(([, , start, stop, attach, release]) => {
    const T = stop - start, down = new Uint8Array(T);  // (T,)
    for (let t = 0; t < T; t++) down[t] = pose[(start + t) * W + C] > 0.5 && pose[(start + t) * W + C + 1] > 0.5;
    for (let t = 1; t < T;) {
      let u = t;
      while (u < T && !down[u]) u++;
      if (u > t && u < T && u - t < MIN_LIFT && down[t - 1]) down.fill(1, t, u);
      t = u + 1;
    }
    const out = [];
    for (const [kind, e] of [['pick', attach], ['place', release]]) {
      if (!(e >= 0 && e < T)) continue;
      let a = e;
      if (!down[e]) {
        while (a < Math.min(T, e + LATE_STEP) && !down[a]) a++;
        if (a >= T || !down[a]) continue;
      }
      while (a > 0 && down[a - 1]) a--;
      let b = Math.max(a, e);
      while (b < T && down[b]) b++;
      out.push({ kind, event: e, step: a, lift: b });
      planted.fill(1, start + a, start + b);
    }
    return out;
  });
  return { clips, planted };
}

const fetchOne = async (base, f) => { const r = await fetch(`${base}/${f}`); if (!r.ok) throw new Error(f); return r.arrayBuffer(); };
// A file over GitHub's 100 MiB limit is exported in parts, <name>.part0.bin and on (cfg.parts, lmm/export_web.py).
async function fetchBin(base, f, parts = 1) {
  if (parts === 1) return fetchOne(base, f);
  const bufs = await Promise.all(Array.from({ length: parts }, (_, k) => fetchOne(base, f.replace('.bin', `.part${k}.bin`))));
  const out = new Uint8Array(bufs.reduce((n, b) => n + b.byteLength, 0));
  let at = 0;
  for (const b of bufs) { out.set(new Uint8Array(b), at); at += b.byteLength; }
  return out.buffer;
}

function fromHalf(half) {
  const out = new Float32Array(half.length);
  const tmp = new Float32Array(1), tmpU = new Uint32Array(tmp.buffer);
  for (let i = 0; i < half.length; i++) {
    const h = half[i], s = (h & 0x8000) << 16, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) out[i] = (s ? -1 : 1) * f * 5.960464477539063e-8;
    else if (e === 31) out[i] = f ? NaN : (s ? -Infinity : Infinity);
    else { tmpU[0] = s | ((e + 112) << 23) | (f << 13); out[i] = tmp[0]; }
  }
  return out;
}

// shared: the folder of mm_pose.bin and mm_loco.bin when the variant shares them with another (lmm/variants.py).
export async function loadMMData(base, cfg, shared = base) {
  const pose = fromHalf(new Uint16Array(await fetchBin(shared, 'mm_pose.bin', cfg.parts?.mm_pose)));
  const db = {};
  for (const name of ['loco', 'box']) {
    const c = cfg.controllers[name];
    const buf = await fetchBin(name === 'loco' ? shared : base, `mm_${name}.bin`, cfg.parts?.[`mm_${name}`]);
    const X = new Float32Array(buf, 0, c.nframes * c.nfeatures);
    const idx = new Int32Array(buf, X.byteLength, c.nframes);
    const ranges = new Int32Array(buf, X.byteLength + idx.byteLength, c.nranges * 2);
    const rangeStop = new Int32Array(c.nframes);
    for (let r = 0; r < c.nranges; r++) for (let i = ranges[2 * r]; i < ranges[2 * r + 1]; i++) rangeStop[i] = ranges[2 * r + 1];
    db[name] = { X, idx, ranges, rangeStop, nfeatures: c.nfeatures };
  }
  db.box.rangeKind = cfg.controllers.box.range_kind;
  db.pick = boxPart(db.box, cfg.controllers.box, 0);
  db.place = boxPart(db.box, cfg.controllers.box, 1);
  return { pose, db };
}

// The pick (kind 0) or place (kind 1) ranges of the box database as a Matching Database of their own. The tag is the
// same for all its frames and for the query, so it never adds cost.
function boxPart(d, c, kind) {
  const n = c.nfeatures, keep = [];
  for (let r = 0; r < c.nranges; r++) if (c.range_kind[r] === kind) keep.push(r);
  const len = keep.reduce((s, r) => s + d.ranges[2 * r + 1] - d.ranges[2 * r], 0);
  const raw = new Float64Array(len * n);  // (len, n)
  const idx = new Int32Array(len), ranges = new Int32Array(2 * keep.length);
  let i = 0;
  keep.forEach((r, k) => {
    ranges[2 * k] = i;
    for (let f = d.ranges[2 * r]; f < d.ranges[2 * r + 1]; f++, i++) {
      idx[i] = d.idx[f];
      for (let j = 0; j < n; j++) raw[i * n + j] = d.X[f * n + j] * c.scale[j] + c.offset[j];
    }
    ranges[2 * k + 1] = i;
  });
  return normalizedDb(raw, idx, ranges, c);
}

// Extra Locomotion frames of lmm/emm_loco.py (dir): the EMM locomotion takes (empty-handed) or the carry spans of EMM's
// box take (emm_carry). Their frames go after the variant's in the Animation Database.
async function loadExtra(dir, cfg, backend) {
  const e = await (await fetch(`${dir}/config.json`)).json();
  if (e.nfeatures !== cfg.controllers.loco.nfeatures || e.pose_width !== cfg.pose_width) throw new Error(`${dir} does not fit this variant`);
  const extra = fromHalf(new Uint16Array(await fetchBin(dir, 'mm_pose.bin')));
  const feats = new Float32Array(await fetchBin(dir, 'mm_loco.bin'));  // (e.nframes, n)
  const first = backend.pose.length / cfg.pose_width;
  const pose = new Float32Array(backend.pose.length + extra.length);
  pose.set(backend.pose);
  pose.set(extra, backend.pose.length);
  backend.pose = pose;
  return { cfg: e, first, feats };
}

// The Locomotion Matching Database of an empty-handed source and a carrying source, each 'nsm' (the variant's
// frames) or 'emm' (the EMM locomotion takes, or the carry spans of EMM's box take). Its name is 'nsm', 'emm',
// 'emmcarry' or 'emmboth'.
export const locoName = (empty, carry) => (empty === 'emm' ? (carry === 'emm' ? 'emmboth' : 'emm') : (carry === 'emm' ? 'emmcarry' : 'nsm'));
export const locoParts = (name) => ({ empty: name === 'emm' || name === 'emmboth' ? 'emm' : 'nsm', carry: name === 'emmcarry' || name === 'emmboth' ? 'emm' : 'nsm' });

function buildLoco(cfg, backend, name) {
  const { empty, carry } = locoParts(name);
  const c = cfg.controllers.loco, n = c.nfeatures, tag = n - 1, d = backend.locos.nsm;
  const parts = [];  // [frame -> Animation Database row, frame -> raw features (n,)]
  // Carrying first, then empty-handed, the order of the 'emm' database before v2.11, so its sums stay the same.
  for (const held of [true, false]) {
    const extra = held ? (carry === 'emm' ? backend.emmCarry : null) : (empty === 'emm' ? backend.emm : null);
    if (extra) {
      for (const [, , start, stop] of extra.cfg.clips.rows) {
        parts.push([stop - start, (i) => extra.first + start + i, (i, j) => extra.feats[(start + i) * n + j]]);
      }
      continue;
    }
    for (let r = 0; r < c.nranges; r++) {
      const a = d.ranges[2 * r], b = d.ranges[2 * r + 1];
      if ((d.X[a * n + tag] * c.scale[tag] + c.offset[tag] > 0.5) !== held) continue;
      parts.push([b - a, (i) => d.idx[a + i], (i, j) => d.X[(a + i) * n + j] * c.scale[j] + c.offset[j]]);
    }
  }
  const len = parts.reduce((s, [m]) => s + m, 0);
  const raw = new Float64Array(len * n);  // (len, n)
  const idx = new Int32Array(len), ranges = new Int32Array(2 * parts.length);
  let i = 0;
  parts.forEach(([m, row, feat], k) => {
    ranges[2 * k] = i;
    for (let f = 0; f < m; f++, i++) {
      idx[i] = row(f);
      for (let j = 0; j < n; j++) raw[i * n + j] = feat(f, j);
    }
    ranges[2 * k + 1] = i;
  });
  backend.locos[name] = normalizedDb(raw, idx, ranges, c);
}

// Loads what the Locomotion Matching Database `name` needs (emmDir: the EMM locomotion takes, carryDir: the EMM carry
// spans) and builds it, normalized on its own frames as normalize_groups does in training.
export async function loadLoco(name, cfg, backend, emmDir, carryDir) {
  const { empty, carry } = locoParts(name);
  if (empty === 'emm' && !backend.emm) backend.emm = await loadExtra(emmDir, cfg, backend);
  if (carry === 'emm' && !backend.emmCarry) backend.emmCarry = await loadExtra(carryDir, cfg, backend);
  if (!backend.locos[name]) buildLoco(cfg, backend, name);
}

// The EMM locomotion takes with the variant's carrying frames ('emm'), as before v2.11.
export const loadEmmLoco = (dir, cfg, backend) => loadLoco('emm', cfg, backend, dir, null);

// A Matching Database of raw features (len, n), normalized on its own frames as in normalize_groups of
// lmm/database.py, with the weights of the variant's database c.
function normalizedDb(raw, idx, ranges, c) {
  const n = c.nfeatures, len = idx.length, rangeStop = new Int32Array(len);
  for (let k = 0; k < ranges.length / 2; k++) rangeStop.fill(ranges[2 * k + 1], ranges[2 * k], ranges[2 * k + 1]);
  const offset = new Float64Array(n), scale = new Float64Array(n);  // (n,)
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let f = 0; f < len; f++) s += raw[f * n + j];
    offset[j] = s / len;
  }
  let j0 = 0;
  c.group_sizes.forEach((size, g) => {
    let std = 0;
    for (let j = j0; j < j0 + size; j++) {
      let s = 0;
      for (let f = 0; f < len; f++) s += (raw[f * n + j] - offset[j]) ** 2;
      std += Math.sqrt(s / len) / size;
    }
    // The tag of a box part is constant, up to float32 rounding.
    scale.fill(std > 1e-6 ? std / c.weights[g] : 1, j0, j0 + size);
    j0 += size;
  });
  const X = new Float32Array(len * n);  // (len, n)
  for (let f = 0; f < len; f++) for (let j = 0; j < n; j++) X[f * n + j] = (raw[f * n + j] - offset[j]) / scale[j];
  return { X, idx, ranges, rangeStop, nfeatures: n, offset, scale };
}

export const variantDefaults = (cfg) => ({ warp: cfg.warp ?? DEFAULTS.warp, leaveMoved: cfg.leave_moved_m !== undefined ? cfg.leave_moved_m : DEFAULTS.leaveMoved,
  rise: cfg.rise ?? DEFAULTS.rise, transitionCost: cfg.transition_cost ?? DEFAULTS.transitionCost });

export class Controller {
  constructor(cfg, backend, opts = {}) {
    this.cfg = cfg;
    this.p = { ...DEFAULTS, ...opts, weights: {}, gains: {} };
    delete this.p.bodies;
    if (opts.uprightTilt === undefined) this.p.uprightTilt = cfg.upright_tilt_deg ?? null;
    Object.assign(this.p, variantDefaults(cfg), opts.warp === undefined ? {} : { warp: opts.warp },
      opts.leaveMoved === undefined ? {} : { leaveMoved: opts.leaveMoved }, opts.rise === undefined ? {} : { rise: opts.rise },
      opts.transitionCost === undefined ? {} : { transitionCost: opts.transitionCost });
    this.torso = torsoChain(opts.bodies);
    this.p.floorBoxZ = cfg.box_size[2] / 2 + this.p.floorBoxLift;
    // A variant whose place clips do not walk in (holosoma-scenebot-place) is put down from closer, and its
    // put-down starts at the first frames of a place clip, as the clip is played in its deploy.
    this.p.placeEngageRadius = cfg.place_engage_radius ?? this.p.forceEngageRadius;
    this.p.placeEntryFrames = cfg.place_entry_frames ?? 0;
    this.dt = 1 / cfg.fps;
    for (const kind in QUERY_DB) this.setWeights(kind, cfg.controllers[QUERY_DB[kind]].weights);
    this.setBackend(backend, true);
  }

  // Exact search only. The stored features carry the database weights, so a new weight w scales the squared
  // error of its features by (w / w0)². The Projector learned the nearest neighbour under w0, so the learned
  // backend ignores the gains.
  setWeights(kind, weights) {
    const c = this.cfg.controllers[QUERY_DB[kind]];
    const gain = new Float64Array(c.nfeatures);  // (nfeatures,)
    let i = 0;
    c.group_sizes.forEach((n, g) => { gain.fill((weights[g] / c.weights[g]) ** 2, i, i + n); i += n; });
    this.p.weights[kind] = weights.slice();
    this.p.gains[kind] = gain;
  }

  // boxEnter, boxLeave or sideMargin of exact search.
  setThreshold(name, value) {
    this.p[name] = { ...this.p[name], mm: value };
  }

  setBackend(backend, reset = false) {
    const keep = this.pose ? { pos: this.pose.rootPos.slice(), rot: this.pose.rootRot.slice(), holding: this.holding } : null;
    this.backend = backend;
    if (reset || !keep) this.reset([0, 0, 0], 0, false);
    else this.reset(keep.pos, q.yaw(keep.rot), keep.holding);
  }

  reset(pos, yaw, holding = false) {
    const rot = q.fromYaw(yaw);
    this.mode = 'loco';
    this.boxTag = null;
    this.holding = holding;
    this.sim = { pos: pos.slice(), vel: [0, 0, 0], acc: [0, 0, 0], rot: rot.slice(), ang: [0, 0, 0] };
    this.desiredRot = rot.slice();
    this.desiredVel = [0, 0, 0];
    this.searchTimer = 0;
    this.forceSearch = true;
    this.holdSwitchTime = 0;
    this.commit = null;
    this.target = null;
    this.boxCost = Infinity;
    this.lastCosts = [];
    this.lastMatch = {};
    this.calledOff = false;
    this.engaged = null;
    this.reapproach = null;
    this.leaveBox = false;
    this.missed = false;
    this.walkIn = false;
    this.semantics = 'none';
    this.events = { grasp: false, release: false };
    this.trajectory = { pos: [pos, pos, pos, pos].map((p) => p.slice()), rot: [rot, rot, rot, rot].map((r) => r.slice()) };

    const b = this.backend;
    const m = this.match('loco', this.standingQuery(holding));
    b.commit(m);
    const p = b.matchPose(m, pos.slice(), rot.slice(), this.dt);
    p.rootPos = pos.slice(); p.rootRot = rot.slice();
    this.curr = p;
    this.pose = copyPose(p);
    this.off = {
      rootPos: [0, 0, 0], rootVel: [0, 0, 0], rootRot: q.id(), rootAng: [0, 0, 0],
      hipPos: [0, 0, 0], hipVel: [0, 0, 0], hipRot: q.id(), hipAng: [0, 0, 0],
      joints: new Float64Array(NDOF), jointVel: new Float64Array(NDOF),
      boxPos: [0, 0, 0], boxVel: [0, 0, 0], boxRot: q.id(), boxAng: [0, 0, 0],
    };
    this.tsrc = { pos: pos.slice(), rot: rot.slice() };
    this.tdst = { pos: pos.slice(), rot: rot.slice() };
  }

  standingQuery(holding) {
    const c = this.backend.ctrl.loco;
    const qy = Float32Array.from(c.offset);
    for (let i = 6; i < NPOSE_FEATURES; i++) qy[i] = 0;
    for (let i = NPOSE_FEATURES; i < NPOSE_FEATURES + 12; i++) qy[i] = 0;
    for (let h = 0; h < 3; h++) qy[NPOSE_FEATURES + 6 + 2 * h] = 1;
    qy[NPOSE_FEATURES + 12] = holding ? 0 : 1;
    qy[NPOSE_FEATURES + 13] = holding ? 1 : 0;
    return qy;
  }

  // ---- inertialization (port of inertialize_pose_transition / _update / _root_adjust) ----

  transition(dst) {
    const P = this.pose, C = this.curr, O = this.off;
    this.tdst = { pos: P.rootPos.slice(), rot: P.rootRot.slice() };
    this.tsrc = { pos: dst.rootPos.slice(), rot: dst.rootRot.slice() };
    const toWorld = (v) => q.mulVec(this.tdst.rot, q.invMulVec(this.tsrc.rot, v));
    const wdv = toWorld(dst.rootVel), wda = toWorld(dst.rootAng);
    O.rootVel = v3.sub(v3.add(P.rootVel, O.rootVel), wdv);
    O.rootAng = v3.sub(v3.add(O.rootAng, P.rootAng), wda);
    O.hipPos = v3.sub(v3.add(C.hipPos, O.hipPos), dst.hipPos);
    O.hipVel = v3.sub(v3.add(C.hipVel, O.hipVel), dst.hipVel);
    O.hipRot = q.abs(q.mul(q.mul(O.hipRot, C.hipRot), q.inv(dst.hipRot)));
    O.hipAng = v3.sub(v3.add(O.hipAng, C.hipAng), dst.hipAng);
    O.boxPos = v3.sub(v3.add(C.boxPos, O.boxPos), dst.boxPos);
    O.boxRot = q.abs(q.mul(q.mul(O.boxRot, C.boxRot), q.inv(dst.boxRot)));
    for (let i = 0; i < NDOF; i++) {
      O.joints[i] = C.joints[i] + O.joints[i] - dst.joints[i];
      O.jointVel[i] = C.jointVel[i] + O.jointVel[i] - dst.jointVel[i];
    }
    this.curr = dst;
  }

  inertializeUpdate() {
    const P = this.pose, C = this.curr, O = this.off, h = this.p.inertializeHalflife, dt = this.dt;
    const wpos = v3.add(q.mulVec(this.tdst.rot, q.invMulVec(this.tsrc.rot, v3.sub(C.rootPos, this.tsrc.pos))), this.tdst.pos);
    const wvel = q.mulVec(this.tdst.rot, q.invMulVec(this.tsrc.rot, C.rootVel));
    const wrot = q.norm(q.mul(this.tdst.rot, q.invMul(this.tsrc.rot, C.rootRot)));
    const wang = q.mulVec(this.tdst.rot, q.invMulVec(this.tsrc.rot, C.rootAng));
    [O.rootPos, O.rootVel] = decaySpringVec(O.rootPos, O.rootVel, h, dt);
    P.rootPos = v3.add(wpos, O.rootPos); P.rootVel = v3.add(wvel, O.rootVel);
    [O.rootRot, O.rootAng] = decaySpringQuat(O.rootRot, O.rootAng, h, dt);
    P.rootRot = q.norm(q.mul(O.rootRot, wrot)); P.rootAng = v3.add(O.rootAng, q.mulVec(O.rootRot, wang));
    [O.hipPos, O.hipVel] = decaySpringVec(O.hipPos, O.hipVel, h, dt);
    P.hipPos = v3.add(C.hipPos, O.hipPos); P.hipVel = v3.add(C.hipVel, O.hipVel);
    [O.hipRot, O.hipAng] = decaySpringQuat(O.hipRot, O.hipAng, h, dt);
    P.hipRot = q.norm(q.mul(O.hipRot, C.hipRot)); P.hipAng = v3.add(O.hipAng, q.mulVec(O.hipRot, C.hipAng));
    [O.boxPos, O.boxVel] = decaySpringVec(O.boxPos, O.boxVel, h, dt);
    P.boxPos = v3.add(C.boxPos, O.boxPos);
    [O.boxRot, O.boxAng] = decaySpringQuat(O.boxRot, O.boxAng, h, dt);
    P.boxRot = q.norm(q.mul(O.boxRot, C.boxRot));
    for (let i = 0; i < NDOF; i++) {
      [O.joints[i], O.jointVel[i]] = decaySpringScalar(O.joints[i], O.jointVel[i], h, dt);
      P.joints[i] = C.joints[i] + O.joints[i];
      P.jointVel[i] = C.jointVel[i] + O.jointVel[i];
    }
    P.contacts = C.contacts.slice();
    P.held = C.held;
  }

  rootAdjust(inputPos, inputRot) {
    const P = this.pose, O = this.off;
    const dpos = v3.sub(inputPos, P.rootPos);
    P.rootPos = v3.add(P.rootPos, dpos);
    this.tdst.pos = v3.add(this.tdst.pos, dpos);
    this.tsrc.pos = v3.add(this.tsrc.pos, q.mulVec(this.tsrc.rot,
      q.invMulVec(this.tdst.rot, v3.sub(v3.sub(P.rootPos, O.rootPos), this.tdst.pos))));
    this.tdst.pos = P.rootPos.slice();
    O.rootPos = [0, 0, 0];
    const drot = q.norm(q.mulInv(inputRot, P.rootRot));
    P.rootRot = q.mul(drot, P.rootRot);
    this.tdst.rot = q.mul(drot, this.tdst.rot);
  }

  // Moves the character, its simulation object and its trajectory by d (x, y), for drift correction
  // in the physical mode.
  shift(d) {
    const P = this.pose;
    this.rootAdjust([P.rootPos[0] + d[0], P.rootPos[1] + d[1], P.rootPos[2]], P.rootRot);
    this.sim.pos = [this.sim.pos[0] + d[0], this.sim.pos[1] + d[1], this.sim.pos[2]];
    for (const p of this.trajectory.pos) { p[0] += d[0]; p[1] += d[1]; }
  }

  // Turns the character, its simulation object and its trajectory by dyaw about the vertical line through
  // pivot (x, y), for the snap-xyyaw* and snap-all modes of the physical mode. Box targets stay in the world.
  turn(dyaw, pivot) {
    const c = Math.cos(dyaw), s = Math.sin(dyaw), r = q.fromYaw(dyaw);
    const pt = (p) => [pivot[0] + c * (p[0] - pivot[0]) - s * (p[1] - pivot[1]), pivot[1] + s * (p[0] - pivot[0]) + c * (p[1] - pivot[1]), p[2]];
    const vec = (v) => [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]];
    const P = this.pose;
    this.rootAdjust(pt(P.rootPos), q.mul(r, P.rootRot));
    P.rootVel = vec(P.rootVel);
    this.off.rootVel = vec(this.off.rootVel);
    Object.assign(this.sim, { pos: pt(this.sim.pos), vel: vec(this.sim.vel), acc: vec(this.sim.acc), rot: q.mul(r, this.sim.rot) });
    this.desiredRot = q.mul(r, this.desiredRot);
    this.desiredVel = vec(this.desiredVel);
    this.trajectory.pos = this.trajectory.pos.map(pt);
    this.trajectory.rot = this.trajectory.rot.map((x) => q.mul(r, x));
  }

  // The snap modes with joints: the pose continues from the robot's angles and velocities of the given joints,
  // and with height from its pelvis height, through the inertialization offsets. qpos (36) and qvel (35) of the
  // robot, joints in the order of the pose.
  matchRobot(qpos, qvel, joints, height) {
    const C = this.curr, O = this.off;
    for (const i of joints) {
      O.joints[i] = qpos[7 + i] - C.joints[i];
      O.jointVel[i] = qvel[6 + i] - C.jointVel[i];
    }
    if (!height) return;
    O.hipPos[2] = qpos[2] - this.pose.rootPos[2] - C.hipPos[2];
    O.hipVel[2] = qvel[2] - C.hipVel[2];
  }

  // ---- simulation object and trajectory (port of the simulation_* / trajectory_* functions) ----

  desiredVelocity(stick, simRot, speed) {
    if (!stick) return [0, 0, 0];
    const local = q.invMulVec(simRot, [stick[0], stick[1], 0]);
    const s = [local[0] * (local[0] > 0 ? speed.fwd : speed.back), local[1] * speed.side, 0];
    return q.mulVec(simRot, s);
  }

  simPositionUpdate(s, desiredVel, halflife, dt, obstacles) {
    const y = (4 * Math.LN2) / (halflife + 1e-5) / 2;
    const j0 = v3.sub(s.vel, desiredVel);
    const j1 = v3.add(s.acc, v3.scale(j0, y));
    const e = Math.exp(-y * dt);
    const prev = s.pos;
    const pos = v3.add(v3.add(v3.add(
      v3.scale(v3.add(v3.scale(j1, -1 / (y * y)), v3.scale(v3.sub(v3.scale(j0, -1), v3.scale(j1, dt)), 1 / y)), e),
      v3.scale(j1, 1 / (y * y))), v3.scale(j0, 1 / y)), v3.add(v3.scale(desiredVel, dt), prev));
    s.vel = v3.add(v3.scale(v3.add(j0, v3.scale(j1, dt)), e), desiredVel);
    s.acc = v3.scale(v3.sub(s.acc, v3.scale(j1, y * dt)), e);
    s.pos = this.collide(pos, obstacles);
  }

  collide(pos, obstacles) {
    let p = pos.slice();
    const r = this.p.collisionRadius;
    for (const o of obstacles) {
      const c = Math.cos(o.yaw), s = Math.sin(o.yaw);
      const dx = p[0] - o.pos[0], dy = p[1] - o.pos[1];
      const lx = c * dx + s * dy, ly = -s * dx + c * dy;
      const hx = o.half[0] + r, hy = o.half[1] + r;
      if (Math.abs(lx) < hx && Math.abs(ly) < hy) {
        let nx = lx, ny = ly;
        if (hx - Math.abs(lx) < hy - Math.abs(ly)) nx = Math.sign(lx || 1) * hx;
        else ny = Math.sign(ly || 1) * hy;
        p = [o.pos[0] + c * nx - s * ny, o.pos[1] + s * nx + c * ny, 0];
      }
    }
    return p;
  }

  // Turn away from boxes that block the straight line to the goal, to the side they are not on.
  steer(from, dir, goalDist, obstacles) {
    let d = dir.slice();
    for (const o of obstacles) {
      const rel = [o.pos[0] - from[0], o.pos[1] - from[1]];
      const dist = Math.hypot(rel[0], rel[1]);
      const along = rel[0] * dir[0] + rel[1] * dir[1];
      if (dist > 1.6 || along < 0 || along > goalDist) continue;
      const lateral = dir[0] * rel[1] - dir[1] * rel[0];
      const clearance = Math.max(o.half[0], o.half[1]) + this.p.collisionRadius + 0.15;
      if (Math.abs(lateral) >= clearance) continue;
      const side = lateral > 0 ? -1 : 1;
      const w = 2 * (1 - Math.abs(lateral) / clearance) * (1 - dist / 1.6);
      d = [d[0] - side * w * dir[1], d[1] + side * w * dir[0]];
    }
    const l = Math.hypot(d[0], d[1]) || 1;
    return [d[0] / l, d[1] / l];
  }

  predictTrajectory(stick, speed, obstacles) {
    const sim = this.sim, dt = 20 * this.dt;
    const rots = [sim.rot.slice()], drot = [this.desiredRot.slice()];
    const dvel = [this.desiredVel.slice()];
    for (let i = 1; i < 4; i++) {
      const dv = this.desiredVelocity(stick, rots[i - 1], speed);
      drot.push(stick ? q.fromYaw(Math.atan2(dv[1], dv[0])) : drot[i - 1]);
      const st = { x: sim.rot.slice(), v: sim.ang.slice() };
      simpleSpringDamperQuat(st, drot[i], this.p.simRotationHalflife, i * dt);
      rots.push(st.x);
      dvel.push(this.desiredVelocity(stick, rots[i], speed));
    }
    const pos = [sim.pos.slice()];
    const s = { pos: sim.pos.slice(), vel: sim.vel.slice(), acc: sim.acc.slice() };
    for (let i = 1; i < 4; i++) {
      this.simPositionUpdate(s, dvel[i], this.p.simVelocityHalflife, dt, obstacles);
      pos.push(s.pos.slice());
    }
    this.trajectory = { pos, rot: rots };
  }

  // ---- queries ----

  poseFeatures() {
    const b = this.backend;
    return b.denormalize(b.active, b.currentFeatures()).subarray(0, NPOSE_FEATURES);
  }

  locoQuery(holding) {
    const qy = new Float32Array(this.backend.ctrl.loco.nfeatures);
    qy.set(this.poseFeatures());
    const P = this.pose;
    for (let h = 0; h < 3; h++) {
      const tp = q.invMulVec(P.rootRot, v3.sub(this.trajectory.pos[h + 1], P.rootPos));
      const td = q.invMulVec(P.rootRot, q.mulVec(this.trajectory.rot[h + 1], [1, 0, 0]));
      qy[NPOSE_FEATURES + 2 * h] = tp[0]; qy[NPOSE_FEATURES + 2 * h + 1] = tp[1];
      qy[NPOSE_FEATURES + 6 + 2 * h] = td[0]; qy[NPOSE_FEATURES + 6 + 2 * h + 1] = td[1];
    }
    qy[NPOSE_FEATURES + 12] = holding ? 0 : 1;
    qy[NPOSE_FEATURES + 13] = holding ? 1 : 0;
    return qy;
  }

  boxQuery(targetPos, targetDir, tag) {
    const qy = new Float32Array(this.backend.ctrl.box.nfeatures);
    qy.set(this.poseFeatures());
    const P = this.pose;
    let tp = q.invMulVec(P.rootRot, [targetPos[0] - P.rootPos[0], targetPos[1] - P.rootPos[1], 0]);
    const r = Math.hypot(tp[0], tp[1]);
    if (r > this.p.boxQueryRadius) tp = v3.scale(tp, this.p.boxQueryRadius / r);
    const td = q.invMulVec(P.rootRot, [targetDir[0], targetDir[1], 0]);
    qy[NPOSE_FEATURES] = tp[0]; qy[NPOSE_FEATURES + 1] = tp[1];
    qy[NPOSE_FEATURES + 2] = td[0]; qy[NPOSE_FEATURES + 3] = td[1];
    qy[NPOSE_FEATURES + 4] = tag === 'pick' ? 1 : 0; qy[NPOSE_FEATURES + 5] = tag === 'place' ? 1 : 0;
    qy[NPOSE_FEATURES + 6] = this.holding ? 0 : 1; qy[NPOSE_FEATURES + 7] = this.holding ? 1 : 0;
    // While bent for the grasp or the release, the hold state comes from the playing animation. The Stepper
    // switches it a few frames before the Decompressor's held output fires the event, and a query with the
    // scene's hold state would project the robot back to a frame before the event.
    const hold = this.animHold();
    if (this.bentForEvent() && hold) { qy[NPOSE_FEATURES + 6] = hold[0]; qy[NPOSE_FEATURES + 7] = hold[1]; }
    return qy;
  }

  // The hold state [empty, holding] of the playing animation, or null outside the box controller.
  animHold() {
    const b = this.backend;
    if (b.active === 'loco') return null;
    const f = b.denormalize(b.active, b.currentFeatures());
    return [f[NPOSE_FEATURES + 6], f[NPOSE_FEATURES + 7]];
  }

  bentForEvent() {
    const ph = this.boxPhase();
    return (ph === 'pick-pre' || ph === 'place-pre') && this.isLow();
  }

  // Where the playing animation believes the box target is, in world coordinates.
  animTarget() {
    const b = this.backend;
    if (b.active === 'loco') return null;
    const f = b.denormalize(b.active, b.currentFeatures());
    const P = this.pose;
    const pos = v3.add(P.rootPos, q.mulVec(P.rootRot, [f[NPOSE_FEATURES], f[NPOSE_FEATURES + 1], 0]));
    const dir = q.mulVec(P.rootRot, [f[NPOSE_FEATURES + 2], f[NPOSE_FEATURES + 3], 0]);
    return { pos, dir };
  }

  // The held box in world coordinates, from the Decompressor's box outputs.
  heldBoxWorld() {
    const P = this.pose;
    return { pos: v3.add(P.rootPos, q.mulVec(P.rootRot, P.boxPos)), rot: q.norm(q.mul(P.rootRot, P.boxRot)) };
  }

  hipHeight() { return this.pose.hipPos[2]; }
  isLow() { return this.hipHeight() < this.p.zLow; }
  afterLastStep() {
    const b = this.backend;
    return this.mode === 'box' && b.kind === 'mm' && b.planted(b.active, b.frame);
  }
  clipSpeed(name, frame = this.backend.frame) {
    const b = this.backend;
    if (b.kind !== 'mm' || name === 'loco') return 1;
    return this.p.pickPlaceSlow === 'step' && !b.planted(name, frame) ? 1 : this.p.pickPlaceSpeed;
  }
  isUpright() { return this.hipHeight() > this.p.zUpright; }
  torsoTilt() { return torsoTilt(this.torso, this.pose.hipRot, (d) => this.pose.joints[d]); }
  risen() { return this.isUpright() && (this.p.uprightTilt === null || this.torsoTilt() < this.p.uprightTilt); }
  // rise 'finish': the playing box clip is at or past its grasp or release.
  rising() {
    const b = this.backend;
    return this.p.rise === 'finish' && this.mode === 'box' && b.kind === 'mm' && b.active !== 'loco' && b.afterEvent(b.active)[b.frame] === 1;
  }

  // State of the box controller: pre (going for the event) or post (rising after it).
  boxPhase() {
    if (this.mode !== 'box') return null;
    if (this.boxTag === 'pick') return this.holding ? 'pick-post' : 'pick-pre';
    return this.holding ? 'place-pre' : 'place-post';
  }

  // The physical mode's pick and place check: the robot is not where the reference bends down, so the pick or
  // place aborts as when its target moves, and the robot rises and approaches again.
  callOff() {
    this.calledOff = true;
    this.forceSearch = true;
  }

  // ---- the frame update ----

  // input: { stick: [x, y] world direction or null, slow: bool, holding: bool,
  //          pickBox: {pos, yaw, id} or null, placeTarget: {pos, yaw} or null,
  //          boxes: floor boxes [{pos, yaw, id}], boxHalf: [x, y], forceSearch: bool }
  update(input) {
    const dt = this.dt, p = this.p, b = this.backend;
    this.holding = input.holding;
    this.events = { grasp: false, release: false };
    const speedScale = input.slow ? p.slowScale : 1;
    const speed = { fwd: p.walkFwd * speedScale, side: p.walkSide * speedScale, back: p.walkBack * speedScale };
    const goal = input.holding ? input.placeTarget : input.pickBox;
    const activity = input.holding ? (input.placeTarget ? 'place' : null) : (input.pickBox ? 'pick' : null);
    const obstacles = input.boxes.filter((o) => !(this.mode === 'box' && input.pickBox && o.id === input.pickBox.id))
      .map((o) => ({ id: o.id, pos: o.pos, yaw: o.yaw, half: input.boxHalf }));

    let stick = input.stick;
    let navigating = false;
    if (!stick && goal && this.mode === 'loco') {
      const d = [goal.pos[0] - this.sim.pos[0], goal.pos[1] - this.sim.pos[1]];
      const dist = Math.hypot(d[0], d[1]);
      if (dist > 0.05) {
        const mag = Math.min(1, Math.max(0.35, (dist - 0.4) / 1.0));
        const dir = this.steer(this.sim.pos, [d[0] / dist, d[1] / dist], dist, obstacles);
        stick = [mag * dir[0], mag * dir[1]];
        navigating = true;
      }
    }
    this.navigating = navigating;

    const dvel = this.desiredVelocity(stick, this.sim.rot, speed);
    const dvelChange = v3.len(v3.sub(dvel, this.desiredVel)) / dt;
    this.desiredVel = dvel;
    if (stick && Math.hypot(stick[0], stick[1]) > 0.01) this.desiredRot = q.fromYaw(Math.atan2(dvel[1], dvel[0]));
    this.predictTrajectory(stick, speed, obstacles);
    if (input.forceSearch) this.forceSearch = true;
    if (!navigating && dvelChange > 50 * 0.25) this.forceSearch = true;

    const sides = goal ? targetSides(goal.pos, goal.yaw, this.cfg.box_sides) : [];
    const phase = this.boxPhase();
    if (this.commit && this.mode === 'loco') this.commit = null;

    const pre = phase === 'pick-pre' || phase === 'place-pre';
    if (!pre || !goal) this.engaged = null;
    else if (!this.engaged || this.engaged.phase !== phase) this.engaged = { phase, pos: goal.pos.slice() };
    if (this.reapproach !== null) {
      this.reapproach += dt;
      if (this.reapproach > p.reapproachTime || !activity) this.reapproach = null;
    }
    this.leaveBox = pre && this.isUpright() && p.leaveMoved !== null && this.engaged &&
      Math.hypot(goal.pos[0] - this.engaged.pos[0], goal.pos[1] - this.engaged.pos[1]) > p.leaveMoved;
    const rising = this.rising();
    this.missed = rising && pre && (this.risen() || b.endOfAnim());
    if (this.leaveBox || this.missed) this.forceSearch = true;
    if (this.walkIn && (!activity || (this.mode === 'box' && (!pre || this.afterLastStep())))) this.walkIn = false;

    // Abort: the target moved, vanished or the user steers while the robot is bent for the event.
    let abort = false;
    if ((phase === 'pick-pre' || phase === 'place-pre') && !rising && !this.isUpright() && (this.commit || this.target)) {
      const t = this.target;
      const c = this.commit || { boxId: phase === 'pick-pre' && input.pickBox ? input.pickBox.id : null,
                                 side: t.side, pos: t.pos.slice(), yaw: t.yaw, dir: t.dir.slice() };
      const want = phase === 'pick-pre' ? 'pick' : 'place';
      abort = this.calledOff || activity !== want || !!input.stick || (input.pickBox && phase === 'pick-pre' && c.boxId !== input.pickBox.id) ||
        Math.hypot(sides[c.side].pos[0] - c.pos[0], sides[c.side].pos[1] - c.pos[1]) > p.movedTolerance ||
        Math.abs(wrapAngle(sides[c.side].yaw - c.yaw)) > p.movedAngle;
      if (abort && !this.commit) this.commit = c;
    }
    this.calledOff = false;
    if (abort) {
      this.boxTag = this.boxTag === 'pick' ? 'place' : 'pick';
      this.forceSearch = true;
    }

    const hold = this.animHold();
    this.holdSwitchTime = this.bentForEvent() && hold && hold[1] > 0.1 && hold[1] < 0.9 ? this.holdSwitchTime + dt : 0;
    const finishing = rising && !this.risen() && !b.endOfAnim();
    const searchDue = !finishing && (this.forceSearch || this.searchTimer <= 0 || (b.kind === 'mm' && b.endOfAnim()));
    if (searchDue) {
      this.search(input, sides, activity, goal, this.forceSearch, obstacles);
      this.searchTimer = p.searchTime / this.clipSpeed(b.active);
      this.forceSearch = false;
    }
    this.searchTimer -= dt;

    const ph = this.boxPhase();
    if ((ph === 'pick-pre' || ph === 'place-pre') && !this.commit && this.target && this.isLow()) {
      this.commit = { boxId: input.pickBox ? input.pickBox.id : null, side: this.target.side,
                      pos: this.target.pos.slice(), yaw: this.target.yaw, dir: this.target.dir.slice() };
    }

    b.step(dt);
    const next = b.currentPose(this.curr.rootPos, this.curr.rootRot, dt, this.clipSpeed(b.active));
    this.curr = next;
    this.inertializeUpdate();

    // Grasp and release come from the Decompressor's held output, but only where its box pose output
    // agrees: over the real box for a grasp, and down at floor level for a release.
    const held = this.curr.held;
    const hb = this.heldBoxWorld();
    if (ph === 'pick-pre' && held >= p.graspOn && input.pickBox &&
        Math.hypot(hb.pos[0] - input.pickBox.pos[0], hb.pos[1] - input.pickBox.pos[1]) < p.graspReach) {
      this.events.grasp = true;
      if (!this.commit && this.target) this.commit = { boxId: input.pickBox.id, side: this.target.side, pos: this.target.pos.slice(), yaw: this.target.yaw, dir: this.target.dir.slice() };
    }
    if (ph === 'place-pre' && held <= p.releaseOff && hb.pos[2] < p.floorBoxZ) {
      this.events.release = true;
      if (!this.commit && this.target) this.commit = { boxId: null, side: this.target.side, pos: this.target.pos.slice(), yaw: this.target.yaw, dir: this.target.dir.slice() };
    }

    if (this.mode === 'loco') {
      this.simPositionUpdate(this.sim, this.desiredVel, p.simVelocityHalflife, dt, obstacles);
      const st = { x: this.sim.rot, v: this.sim.ang };
      simpleSpringDamperQuat(st, this.desiredRot, p.simRotationHalflife, dt);
      this.sim.rot = st.x; this.sim.ang = st.v;
      this.adjustToSimulation();
    } else {
      if ((ph === 'pick-pre' || ph === 'place-pre') && !((p.warp === 'step' || (p.warp === 'pickstep' && ph === 'pick-pre')) && this.afterLastStep())) this.warpToTarget();
      this.sim.pos = this.pose.rootPos.slice();
      this.sim.vel = this.pose.rootVel.slice();
      this.sim.acc = [0, 0, 0];
      this.sim.rot = this.pose.rootRot.slice();
      this.sim.ang = [0, 0, 0];
      this.desiredRot = this.pose.rootRot.slice();
    }
    this.semantics = this.mode === 'loco' ? (navigating ? 'navigate' : 'locomotion') : this.boxPhase();
  }

  // A side is usable when the straight walk to where the robot stands for it (standOff behind the target) stays
  // clear of the other floor boxes. The box controller walks straight at the target, only locomotion steers.
  approachClear(from, side, obstacles) {
    const p = this.p;
    const stand = [side.pos[0] - p.standOff * side.dir[0], side.pos[1] - p.standOff * side.dir[1]];
    const d = [stand[0] - from[0], stand[1] - from[1]], len2 = d[0] * d[0] + d[1] * d[1] || 1e-9;
    return obstacles.every((o) => {
      const t = Math.max(0, Math.min(1, ((o.pos[0] - from[0]) * d[0] + (o.pos[1] - from[1]) * d[1]) / len2));
      const gap = Math.hypot(from[0] + t * d[0] - o.pos[0], from[1] + t * d[1] - o.pos[1]);
      return gap > Math.hypot(o.half[0], o.half[1]) + p.approachClearance;
    });
  }

  // Exact search only: the Locomotion Matching Database, 'nsm', 'emm', 'emmcarry' or 'emmboth' (loadLoco). Its frame index means nothing
  // in the other database, so a robot in Locomotion is matched again at once.
  setLoco(name) {
    const b = this.backend;
    if (!b.locos?.[name] || b.loco === name) return;
    const qy = b.active === 'loco' ? this.locoQuery(this.holding) : null;
    b.useLoco(name);
    this.p.loco = name;
    if (!qy) return;
    const m = this.match('loco', qy);
    this.transition(b.matchPose(m, this.curr.rootPos, this.curr.rootRot, this.dt));
    b.commit(m);
  }

  // The learned backend has one box Projector, so only exact search can use different box databases.
  dbOf(kind) {
    return kind !== 'loco' && this.p.boxDb === 'different' && this.backend.db?.[kind] ? kind : QUERY_DB[kind];
  }

  match(kind, query, entry = 0) {
    const b = this.backend, name = this.dbOf(kind), qn = b.normalize(name, query), gain = this.p.gains[kind];
    const sbPre = kind !== 'loco' && this.cfg.scenebot_entry_frames && b.sceneBotPre && b.active === name && b.sceneBotPre(name)[b.frame] === 1;
    const mask = kind !== 'loco' && this.p.boxEntry === 'walk' && b.noEntry
      ? (sbPre ? b.beforeEventMask(name) : this.walkIn ? b.walkInMask(name) : b.noEntry(name)) : null;
    return { ...b.match(name, qn, gain, undefined, undefined, entry, mask, this.p.transitionCost), kind, qn, gain };
  }

  search(input, sides, activity, goal, forced, obstacles) {
    const b = this.backend, p = this.p, P = this.pose;
    const enter = p.boxEnter[b.kind], leave = p.boxLeave[b.kind], margin = p.sideMargin[b.kind];
    this.lastCosts = [];
    let boxMatch = null;
    const phase = this.boxPhase();

    if (phase === 'pick-post' || phase === 'place-post') {
      // Rising after the event (or after an abort): the target stays where the event happened.
      const c = this.commit || this.target;
      if (c && !this.risen()) {
        boxMatch = this.match(this.boxTag, this.boxQuery(c.pos, c.dir, this.boxTag));
      }
    } else if (activity && sides.length) {
      const near = Math.hypot(goal.pos[0] - P.rootPos[0], goal.pos[1] - P.rootPos[1]);
      if (this.mode === 'box' || near < p.engageRadius) {
        const locked = this.mode === 'box' && this.target && (this.isLow() || this.commit);
        const others = obstacles.filter((o) => !input.pickBox || o.id !== input.pickBox.id);
        let cands = locked ? [sides[this.target.side]] : sides.filter((s) => this.approachClear(P.rootPos, s, others));
        // A target pressed against another box blocks every side, so the robot takes any side once it is there.
        if (!cands.length && near < p.approachFallback) cands = sides;
        let best = null;
        const entry = this.mode === 'loco' && activity === 'place' ? p.placeEntryFrames : 0;
        for (const s of cands) {
          const m = this.match(activity, this.boxQuery(s.pos, s.dir, activity), entry);
          this.lastCosts.push(m.cost);
          const bonus = (x) => (this.target && x.side === this.target.side ? margin : 0);
          if (!best || m.cost - bonus(s) < best.m.cost - bonus(best.s)) best = { s, m };
        }
        if (best) {
          boxMatch = best.m;
          this.target = best.s;
        }
      }
    }
    this.boxCost = boxMatch ? boxMatch.cost : Infinity;
    if (boxMatch) this.lastMatch[boxMatch.kind] = boxMatch;

    let useBox = false;
    const engage = activity === 'place' ? p.placeEngageRadius : p.forceEngageRadius;
    if (this.leaveBox || this.missed) {
      this.reapproach = 0;
      this.engaged = null;
      this.walkIn = true;
    } else if (this.mode === 'box') {
      if (phase === 'pick-post' || phase === 'place-post') useBox = !!boxMatch;
      else if (!activity || input.stick) useBox = !this.isUpright() && !!boxMatch;
      else {
        // Near the target the robot stays in the box controller even at a high cost, as it entered it.
        const dist = Math.hypot(goal.pos[0] - P.rootPos[0], goal.pos[1] - P.rootPos[1]);
        useBox = !!boxMatch && !(this.isUpright() && this.boxCost > leave && dist > engage + 0.3);
      }
    } else if (boxMatch && !input.stick) {
      const near = Math.hypot(goal.pos[0] - P.rootPos[0], goal.pos[1] - P.rootPos[1]) < engage;
      useBox = this.boxCost < enter || (near && this.reapproach === null);
      if (useBox) {
        this.boxTag = activity;
        this.reapproach = null;
      }
    }

    let m;
    if (useBox) {
      m = boxMatch;
      this.mode = 'box';
    } else {
      if (this.mode === 'box') { this.commit = null; this.boxTag = null; }
      this.mode = 'loco';
      if (!activity) this.target = null;
      m = this.match('loco', this.locoQuery(input.holding));
      this.lastMatch.loco = m;
    }
    if (b.kind === 'mm' && m.same) return;
    // The transition cost of projector_evaluate in lmm.h. Near the box the query hardly changes, and the
    // Projector maps a still robot back to the first still frame before the pick or the rise every 0.1 s,
    // so the robot would never get going. A projection close to the current features is skipped unless
    // the search was forced, for example because the box or the marker moved.
    if (b.kind === 'lmm' && useBox && !forced && b.active !== 'loco') {
      let d = 0;
      for (let i = 0; i < m.features.length; i++) d += (m.features[i] - b.features[i]) ** 2;
      if (Math.sqrt(d) < p.boxTransitionCost) return;
      // The Projector only knows one-hot hold states, so while the Stepper switches the hold state during the
      // grasp or the release, any projection sends the robot back to a frame before the event.
      if (this.holdSwitchTime > 0 && this.holdSwitchTime < p.holdSwitchMax) return;
    }
    const dst = b.matchPose(m, this.curr.rootPos, this.curr.rootRot, this.dt, this.clipSpeed(m.name, m.frame));
    this.transition(dst);
    b.commit(m);
  }

  adjustToSimulation() {
    const p = this.p, P = this.pose, dt = this.dt, sim = this.sim;
    let adjPos = v3.scale(v3.sub(sim.pos, P.rootPos), dampFactor(p.adjustPosHalflife, dt));
    const maxLen = p.adjustPosMaxRatio * v3.len(P.rootVel) * dt;
    if (v3.len(adjPos) > maxLen) adjPos = v3.scale(v3.norm(adjPos), maxLen);
    let pos = v3.add(adjPos, P.rootPos);
    const diff = q.abs(q.norm(q.mulInv(sim.rot, P.rootRot)));
    let adjRot = q.slerpShortestApprox(q.id(), diff, dampFactor(p.adjustRotHalflife, dt));
    const maxAng = p.adjustRotMaxRatio * v3.len(P.rootAng) * dt;
    const aa = q.toScaledAngleAxis(adjRot);
    if (v3.len(aa) > maxAng) adjRot = q.fromScaledAngleAxis(v3.scale(v3.norm(aa), maxAng));
    let rot = q.mul(adjRot, P.rootRot);
    if (v3.len(v3.sub(pos, sim.pos)) > p.clampMaxDistance) pos = v3.add(v3.scale(v3.norm(v3.sub(pos, sim.pos)), p.clampMaxDistance), sim.pos);
    if (q.angleBetween(rot, sim.rot) > p.clampMaxAngle) {
      const d = q.abs(q.mulInv(rot, sim.rot));
      const ang = Math.max(-p.clampMaxAngle, Math.min(p.clampMaxAngle, 2 * Math.acos(Math.min(1, d[0]))));
      const ax = v3.norm([d[1], d[2], d[3]]);
      rot = q.mul(q.fromAngleAxis(ang, ax), sim.rot);
    }
    this.rootAdjust(pos, rot);
  }

  // Trajectory warping: move the root so the animation's box target lands on the real one.
  warpToTarget() {
    if (!this.target) return;
    const a = this.animTarget();
    if (!a) return;
    const p = this.p, P = this.pose, dt = this.dt;
    const t = this.target;
    const e = [t.pos[0] - a.pos[0], t.pos[1] - a.pos[1], 0];
    const yawErr = wrapAngle(t.yaw - Math.atan2(a.dir[1], a.dir[0]));
    if (v3.len(e) > p.warpMaxDist || Math.abs(yawErr) > p.warpMaxAngle) return;
    const hipWorldVel = q.mulVec(P.rootRot, P.hipVel);
    const speed = Math.hypot(P.rootVel[0], P.rootVel[1]) + Math.hypot(hipWorldVel[0], hipWorldVel[1]) + Math.abs(hipWorldVel[2]);
    const k = dampFactor(p.warpHalflife, dt);
    let dpos = v3.scale(e, k);
    const maxLen = p.warpMaxRatio * speed * dt;
    if (v3.len(dpos) > maxLen) dpos = v3.scale(v3.norm(dpos), maxLen);
    let dyaw = yawErr * k;
    const maxYaw = p.warpMaxRatio * (Math.abs(P.rootAng[2]) + speed * 1.5) * dt;
    dyaw = Math.max(-maxYaw, Math.min(maxYaw, dyaw));
    // Rotate about the target so the yaw correction does not move it.
    const rq = q.fromYaw(dyaw);
    const rel = v3.sub(P.rootPos, [a.pos[0], a.pos[1], 0]);
    const newPos = v3.add(v3.add([a.pos[0], a.pos[1], 0], q.mulVec(rq, rel)), dpos);
    this.rootAdjust([newPos[0], newPos[1], 0], q.norm(q.mul(rq, P.rootRot)));
  }

  renderState() {
    const P = this.pose;
    const hipWorld = v3.add(P.rootPos, q.mulVec(P.rootRot, P.hipPos));
    const hipRotWorld = q.mul(P.rootRot, P.hipRot);
    return { hipPos: hipWorld, hipRot: hipRotWorld, joints: P.joints, root: P.rootPos, rootRot: P.rootRot, contacts: P.contacts };
  }
}
