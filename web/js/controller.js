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
  // The cost thresholds of a pick and of a place, for each backend (thresholds).
  boxEnter: { pick: { lmm: 2.5, mm: 2.5 }, place: { lmm: 2.5, mm: 2.5 } },
  boxLeave: { pick: { lmm: 4.0, mm: 4.0 }, place: { lmm: 4.0, mm: 4.0 } },
  // Locomotion hands a pick to the box controller within this distance of the box even at a cost above boxEnter, and a
  // place within placeEngageRadius (variantDefaults) of the marker. 0: only by the cost.
  forceEngageRadius: 1.1,
  // Locomotion hands a pick or a place to the box controller, by the cost or by the distance above, only within this
  // distance of the box or marker centre. null: anywhere the box Matching Database is searched (engageRadius).
  takeOverRadius: 1.0,
  sideMargin: { pick: { lmm: 0.25, mm: 0.25 }, place: { lmm: 0.25, mm: 0.25 } },
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
  // back to Locomotion, which may not take the box controller again by the distance (forceEngageRadius,
  // placeEngageRadius) for reapproachTime. Until the clip that takes it again reaches its last step, the search jumps
  // only to frames before a last step (walkInMask).
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
  // Exact search only, in the variants whose Box Matching Database has the SceneBot pick (sceneBotStances). 'stance':
  // once the search has chosen a side of the box or the marker, Locomotion walks to where the SceneBot pick or place
  // of that side starts (the captured or the mirrored one, whichever is nearer, switching only for one stanceSwitch
  // nearer) along the axis of that side (axisPoint), so it arrives facing the box. 'arrive': the same, and within
  // arriveReach of the stance the robot turns to its facing and the query's last trajectory point, 1 s ahead, is the
  // stance (approach). 'centre': straight to the box or marker centre.
  locoTarget: 'arrive', stanceSwitch: 0.05, axisLead: 1.0, axisLook: 0.5, arriveReach: 0.75,
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
  // The pick and place motions take the box (a marker for a place) as this far (m) farther from the robot than the
  // data has it, along the box X axis: the box of a query is moved back by it, and what a clip says about its box
  // (animTarget, heldBoxWorld, the SceneBot stances) is moved ahead by it. The joints move as in the data, so the
  // robot stands that much farther from the box, the hands close that much nearer to its near edge, and the box is
  // carried and put down that much farther ahead of the hands. 0: as the data has it.
  boxAhead: 0,
};

const NO_EDIT = [0, 0, 0];
const WEDGE_IN = 15 * Math.PI / 180;
const GROUP_MIDDLE = 0.1;
const turned = (d, a) => [Math.cos(a) * d[0] - Math.sin(a) * d[1], Math.sin(a) * d[0] + Math.cos(a) * d[1]];

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
export const QUERY_DB = { loco: 'loco', arrive: 'loco', pick: 'box', place: 'box' };
export const isBoxQuery = (kind) => QUERY_DB[kind] === 'box';
const FSM_HISTORY = 24;

// The group that the page adds to the Locomotion Matching Database when it loads it (addTrajectoryVelocity): the
// velocity of the root at the last trajectory point, 1 s ahead, in the root frame of the frame. weight is the weight
// its features are normalized with. arrive is the weight of the arrive query. The Locomotion query has it at 0, so
// Locomotion searches as without the group.
export const TRAJ_VEL = { group: 'traj_vel', size: 2, weight: 1, arrive: 0 };
// The arrive state's own weights, in this order: trajectory position at 20, 40 and 60 frames, trajectory direction at
// the same points, then trajectory velocity at 60 frames. The other Locomotion features keep the Locomotion weights.
export const ARRIVE_GROUPS = ['traj_pos', 'traj_dir'];
export function queryWeights(cfg, kind) {
  const c = cfg.controllers[QUERY_DB[kind]], vel = c.groups.indexOf(TRAJ_VEL.group);
  if (kind !== 'arrive') return c.weights.map((w, g) => (g === vel ? 0 : w));
  return [...ARRIVE_GROUPS.flatMap((g) => Array(3).fill(c.weights[c.groups.indexOf(g)])), ...(vel >= 0 ? [TRAJ_VEL.arrive] : [])];
}
// Weights given without the trajectory velocity (links and commands of before v3.4.1) get its default.
export function fillWeights(cfg, kind, w) {
  const def = queryWeights(cfg, kind);
  return w.length === def.length - 1 && !isBoxQuery(kind) ? [...w, def[def.length - 1]] : w;
}

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
  // The cost of the frame that plays now against a normalized query of its database, with the weights of the search.
  // null when another database plays.
  playCost(name, qn, gain) {
    if (name !== this.active) return null;
    const f = this.currentFeatures();
    let c = 0;
    for (let j = 0; j < f.length; j++) c += gain[j] * (qn[j] - f[j]) ** 2;
    return Math.sqrt(c);
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
    this.stops = data.stops;  // per row of the variant's Animation Database: the row after its clip
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
  // [x, y, yaw] of the root where the SceneBot pick and place start, in the frame of the box side they work on (x
  // away from the robot): {pick: [...], place: [...]}, one per captured or mirrored clip that the Box Matching
  // Database indexes, or null without the SceneBot pick. A pick is measured to the box on the floor at its start, a
  // place to the box at its release.
  sceneBotStances() {
    const { names, rows } = this.cfg.clips, d = this.db.box, L = this.L;
    const pick = names.indexOf('SceneBotPick'), place = names.indexOf('SceneBotPlace');
    if (pick < 0 || !d) return null;
    const indexed = (start, stop) => {
      for (let r = 0; r < d.ranges.length / 2; r++) {
        if (d.idx[d.ranges[2 * r]] < stop && d.idx[d.ranges[2 * r + 1] - 1] >= start) return true;
      }
      return false;
    };
    const at = (g) => this.pose.subarray(g * this.width, (g + 1) * this.width);
    const out = { pick: [], place: [] };
    for (const [n, , start, stop, , release] of rows) {
      if ((n !== pick && n !== place) || !indexed(start, stop)) continue;
      const e = n === pick ? 0 : release;
      let px = 0, py = 0, yaw = 0;
      for (let g = start; g < start + e; g++) {
        const r = at(g), dx = r[L.root_delta[0]], dy = r[L.root_delta[0] + 1];
        px += Math.cos(yaw) * dx - Math.sin(yaw) * dy;
        py += Math.sin(yaw) * dx + Math.cos(yaw) * dy;
        yaw += r[L.root_delta[0] + 2];
      }
      const r = at(start + e), bx = r[L.box_pos[0]], by = r[L.box_pos[0] + 1];
      const cx = px + Math.cos(yaw) * bx - Math.sin(yaw) * by, cy = py + Math.sin(yaw) * bx + Math.cos(yaw) * by;
      const byaw = yaw + q.yaw(Array.from(r.subarray(L.box_quat[0], L.box_quat[0] + 4)));
      const c = Math.cos(byaw), s = Math.sin(byaw);
      out[n === pick ? 'pick' : 'place'].push([-(c * cx + s * cy), s * cx - c * cy, wrapAngle(-byaw)]);
    }
    return out;
  }
  // A frame of a box Matching Database between the last step before its grasp or release and the first step after.
  planted(name, frame) {
    return name !== 'loco' && this.steps.planted[this.db[name].idx[frame]] === 1;
  }
  // How much higher the pelvis of the playing frame is than in the data, where the pick motion is edited (edits,
  // pickmotion.js).
  editLift() {
    if (!this.edits || !this.active || this.active === 'loco') return 0;
    return this.edits.lift(this.db[this.active].idx[this.frame]);
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
    return { name: this.cfg.clips.names[n], row: lo, start, stop, event: (d.rangeKind ? d.rangeKind[r] : name === 'place') ? release : attach };
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
  // Per frame of a box Matching Database: 1 at or past the last step before its clip's grasp or release (lastSteps), and
  // on every frame of a clip that has no last step.
  fromLastStep(name) {
    this.fromLastStepOf ??= {};
    if (this.fromLastStepOf[name]) return this.fromLastStepOf[name];
    const d = this.db[name], out = new Uint8Array(d.idx.length);
    for (let r = 0; r < d.ranges.length / 2; r++) {
      const { row, start, stop, event } = this.rangeClip(name, r);
      const s = d.idx[d.ranges[2 * r]] < stop ? this.steps.clips[row].find((x) => x.event === event) : null;
      for (let i = d.ranges[2 * r]; i < d.ranges[2 * r + 1]; i++) if (!s || d.idx[i] - start >= s.step) out[i] = 1;
    }
    return (this.fromLastStepOf[name] = out);
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
    const c = cfg.controllers[name], n = c.stored ?? c.nfeatures;
    const buf = await fetchBin(name === 'loco' ? shared : base, `mm_${name}.bin`, cfg.parts?.[`mm_${name}`]);
    const X = new Float32Array(buf, 0, c.nframes * n);
    const idx = new Int32Array(buf, X.byteLength, c.nframes);
    const ranges = new Int32Array(buf, X.byteLength + idx.byteLength, c.nranges * 2);
    const rangeStop = new Int32Array(c.nframes);
    for (let r = 0; r < c.nranges; r++) for (let i = ranges[2 * r]; i < ranges[2 * r + 1]; i++) rangeStop[i] = ranges[2 * r + 1];
    db[name] = { X, idx, ranges, rangeStop, nfeatures: n };
  }
  const stops = clipStops(pose.length / cfg.pose_width, cfg.clips.rows);
  db.loco = addTrajectoryVelocity(cfg, pose, stops, db.loco);
  db.box.rangeKind = cfg.controllers.box.range_kind;
  db.pick = boxPart(db.box, cfg.controllers.box, 0);
  db.place = boxPart(db.box, cfg.controllers.box, 1);
  return { pose, db, stops };
}

// Per row of an Animation Database: the row after the last one of its clip. rows: [name, mirror, start, stop, ...].
function clipStops(nrows, rows) {
  const out = new Int32Array(nrows);  // (nrows,)
  for (let g = 0; g < nrows; g++) out[g] = g + 1;
  for (const [, , start, stop] of rows) out.fill(stop, start, stop);
  return out;
}

// The velocity of the root `ahead` rows after row g, or at the last row of its clip (stop - 1), in the root frame of
// row g: the stored root velocity of that row, turned by the yaw the root turns until then.
function futureVelocity(pose, width, L, g, stop, ahead) {
  const g1 = Math.min(g + ahead, stop - 1);
  let yaw = 0;
  for (let k = g; k < g1; k++) yaw += pose[k * width + L.root_delta[0] + 2];
  const vx = pose[g1 * width + L.root_vel[0]], vy = pose[g1 * width + L.root_vel[0] + 1];
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return [c * vx - s * vy, s * vx + c * vy];
}

// The Locomotion Matching Database of the files (d) with the group TRAJ_VEL after its features, normalized on its
// frames as normalize_groups of lmm/database.py does. The stored features keep their values. cfg.controllers.loco gets
// the group too, and stored, the number of features in the files.
function addTrajectoryVelocity(cfg, pose, stops, d) {
  const c = cfg.controllers.loco, n0 = d.nfeatures, n = n0 + TRAJ_VEL.size, len = d.idx.length;
  const ahead = cfg.horizons[cfg.horizons.length - 1];
  const raw = new Float64Array(len * TRAJ_VEL.size);  // (len, 2)
  const offset = [0, 0];
  for (let i = 0; i < len; i++) {
    const v = futureVelocity(pose, cfg.pose_width, cfg.pose_layout, d.idx[i], stops[d.idx[i]], ahead);
    raw[2 * i] = v[0]; raw[2 * i + 1] = v[1];
    offset[0] += v[0] / len; offset[1] += v[1] / len;
  }
  let std = 0;
  for (let j = 0; j < 2; j++) {
    let sum = 0;
    for (let i = 0; i < len; i++) sum += (raw[2 * i + j] - offset[j]) ** 2;
    std += Math.sqrt(sum / len) / 2;
  }
  const scale = std > 1e-6 ? std / TRAJ_VEL.weight : 1;
  const X = new Float32Array(len * n);  // (len, n)
  for (let i = 0; i < len; i++) {
    X.set(d.X.subarray(i * n0, (i + 1) * n0), i * n);
    for (let j = 0; j < 2; j++) X[i * n + n0 + j] = (raw[2 * i + j] - offset[j]) / scale;
  }
  if (!c.groups.includes(TRAJ_VEL.group)) {
    Object.assign(c, { stored: n0, nfeatures: n, groups: [...c.groups, TRAJ_VEL.group], group_sizes: [...c.group_sizes, TRAJ_VEL.size],
      weights: [...c.weights, TRAJ_VEL.weight], offset: [...c.offset, ...offset], scale: [...c.scale, scale, scale] });
  }
  return { ...d, X, nfeatures: n };
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
  if (e.nfeatures !== cfg.controllers.loco.stored || e.pose_width !== cfg.pose_width) throw new Error(`${dir} does not fit this variant`);
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
  const c = cfg.controllers.loco, n = c.nfeatures, n0 = c.stored, tag = n0 - 1, d = backend.locos.nsm;
  const ahead = cfg.horizons[cfg.horizons.length - 1];
  // [frames, frame -> Animation Database row, (frame, j < n0) -> stored feature, frame -> the row after its clip]
  const parts = [];
  // Carrying first, then empty-handed, the order of the 'emm' database before v2.11, so its sums stay the same.
  for (const held of [true, false]) {
    const extra = held ? (carry === 'emm' ? backend.emmCarry : null) : (empty === 'emm' ? backend.emm : null);
    if (extra) {
      for (const [, , start, stop] of extra.cfg.clips.rows) {
        parts.push([stop - start, (i) => extra.first + start + i, (i, j) => extra.feats[(start + i) * n0 + j], () => extra.first + stop]);
      }
      continue;
    }
    for (let r = 0; r < c.nranges; r++) {
      const a = d.ranges[2 * r], b = d.ranges[2 * r + 1];
      if ((d.X[a * n + tag] * c.scale[tag] + c.offset[tag] > 0.5) !== held) continue;
      parts.push([b - a, (i) => d.idx[a + i], (i, j) => d.X[(a + i) * n + j] * c.scale[j] + c.offset[j], (i) => backend.stops[d.idx[a + i]]]);
    }
  }
  const len = parts.reduce((s, [m]) => s + m, 0);
  const raw = new Float64Array(len * n);  // (len, n)
  const idx = new Int32Array(len), ranges = new Int32Array(2 * parts.length);
  let i = 0;
  parts.forEach(([m, row, feat, stop], k) => {
    ranges[2 * k] = i;
    for (let f = 0; f < m; f++, i++) {
      idx[i] = row(f);
      for (let j = 0; j < n0; j++) raw[i * n + j] = feat(f, j);
      raw.set(futureVelocity(backend.pose, cfg.pose_width, cfg.pose_layout, idx[i], stop(f), ahead), i * n + n0);
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

// A variant whose place clips do not walk in (holosoma-scenebot-place, sbplace) is put down from closer.
export const variantDefaults = (cfg) => ({ warp: cfg.warp ?? DEFAULTS.warp, leaveMoved: cfg.leave_moved_m !== undefined ? cfg.leave_moved_m : DEFAULTS.leaveMoved,
  rise: cfg.rise ?? DEFAULTS.rise, transitionCost: cfg.transition_cost ?? DEFAULTS.transitionCost,
  placeEngageRadius: cfg.place_engage_radius ?? DEFAULTS.forceEngageRadius });

export class Controller {
  constructor(cfg, backend, opts = {}) {
    this.cfg = cfg;
    this.p = { ...DEFAULTS, ...opts, weights: {}, gains: {} };
    delete this.p.bodies;
    if (opts.uprightTilt === undefined) this.p.uprightTilt = cfg.upright_tilt_deg ?? null;
    Object.assign(this.p, variantDefaults(cfg), opts.warp === undefined ? {} : { warp: opts.warp },
      opts.leaveMoved === undefined ? {} : { leaveMoved: opts.leaveMoved }, opts.rise === undefined ? {} : { rise: opts.rise },
      opts.transitionCost === undefined ? {} : { transitionCost: opts.transitionCost },
      opts.placeEngageRadius === undefined ? {} : { placeEngageRadius: opts.placeEngageRadius });
    this.torso = torsoChain(opts.bodies);
    this.p.floorBoxZ = cfg.box_size[2] / 2 + this.p.floorBoxLift;
    // [forward, left, yaw] of the box at rest of the pick and the place motion that play, from the data's
    // (PickMotion.offsets): the robot takes a target for the data's box that far back.
    this.boxEdit = { pick: NO_EDIT, place: NO_EDIT };
    // A variant whose place clips do not walk in starts its put-down at the first frames of a place clip, as the clip
    // is played in its deploy.
    this.p.placeEntryFrames = cfg.place_entry_frames ?? 0;
    this.dt = 1 / cfg.fps;
    for (const kind in QUERY_DB) this.setWeights(kind, queryWeights(cfg, kind));
    this.setBackend(backend, true);
  }

  // Exact search only. The stored features carry the database weights, so a new weight w scales the squared
  // error of its features by (w / w0)². The Projector learned the nearest neighbour under w0, so the learned
  // backend ignores the gains.
  setWeights(kind, weights) {
    this.p.weights[kind] = weights.slice();
    if (kind === 'arrive') {
      this.p.gains.arrive = this.arriveGains();
      return;
    }
    const c = this.cfg.controllers[QUERY_DB[kind]];
    const gain = new Float64Array(c.nfeatures);  // (nfeatures,)
    let i = 0;
    c.group_sizes.forEach((n, g) => { gain.fill((weights[g] / c.weights[g]) ** 2, i, i + n); i += n; });
    this.p.gains[kind] = gain;
    if (kind === 'loco' && this.p.weights.arrive) this.p.gains.arrive = this.arriveGains();
  }

  // The Locomotion gains, with the arrive weights on the x and y features of each trajectory point.
  arriveGains() {
    const c = this.cfg.controllers.loco, w = this.p.weights.arrive, gain = this.p.gains.loco.slice();  // (nfeatures,)
    let start = 0;
    c.groups.forEach((g, k) => {
      const j = ARRIVE_GROUPS.indexOf(g);
      for (let h = 0; j >= 0 && h < 3; h++) gain.fill((w[3 * j + h] / c.weights[k]) ** 2, start + 2 * h, start + 2 * h + 2);
      if (g === TRAJ_VEL.group) gain.fill((w[3 * ARRIVE_GROUPS.length] / c.weights[k]) ** 2, start, start + c.group_sizes[k]);
      start += c.group_sizes[k];
    });
    return gain;
  }

  // boxEnter, boxLeave or sideMargin of exact search, for a pick or a place.
  setThreshold(name, act, value) {
    this.p[name] = { ...this.p[name], [act]: { ...this.p[name][act], mm: value } };
  }
  thresholds(act) {
    const p = this.p, k = this.backend.kind;
    return { enter: p.boxEnter[act][k], leave: p.boxLeave[act][k], margin: p.sideMargin[act][k] };
  }

  setBackend(backend, reset = false) {
    const keep = this.pose ? { pos: this.pose.rootPos.slice(), rot: this.pose.rootRot.slice(), holding: this.holding } : null;
    this.backend = backend;
    this.stances = backend.kind === 'mm' ? backend.sceneBotStances() : null;
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
    this.navSide = null;
    this.stance = null;
    this.nav = null;
    this.boxCost = Infinity;
    this.lastCosts = [];
    this.lastMatch = {};
    this.matchAt = {};
    this.searchAt = -Infinity;
    this.finishing = false;
    this.calledOff = false;
    this.engaged = null;
    this.reapproach = null;
    this.leaveBox = false;
    this.missed = false;
    this.walkIn = false;
    this.arriving = false;
    this.semantics = 'none';
    // For the page only (trace, status): the controller's time, the cause noted during an update for a change of state,
    // and the state machine's record. Nothing the controller does reads them.
    this.clock = 0;
    this.why = null;
    this.lastEvent = null;
    this.activity = null;
    this.goalDist = null;
    this.fsm = { state: null, since: 0, history: [] };
    this.events = { grasp: false, release: false };
    this.trajectory = { pos: [pos, pos, pos, pos].map((p) => p.slice()), rot: [rot, rot, rot, rot].map((r) => r.slice()), vel: [0, 0, 0] };

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
    for (let i = NPOSE_FEATURES + 14; i < qy.length; i++) qy[i] = 0;
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
    s.pos = this.collide(pos, obstacles, prev);
  }

  // How deep p is in obstacle o grown by the collision radius: { lx, ly, hx, hy, depth }, depth <= 0 outside.
  inObstacle(p, o) {
    const r = this.p.collisionRadius, c = Math.cos(o.yaw), s = Math.sin(o.yaw);
    const dx = p[0] - o.pos[0], dy = p[1] - o.pos[1];
    const lx = c * dx + s * dy, ly = -s * dx + c * dy, hx = o.half[0] + r, hy = o.half[1] + r;
    return { lx, ly, hx, hy, depth: Math.min(hx - Math.abs(lx), hy - Math.abs(ly)) };
  }

  // from: the position before the move. A robot in or on two obstacles at once (it stands between two boxes closer than
  // twice the collision radius) is pushed out of one into the other, and could not move. There it moves without the
  // part of the move that heads more than WEDGE_IN into either, toward its centre.
  collide(pos, obstacles, from = null) {
    const wedged = from ? obstacles.filter((o) => this.inObstacle(from, o).depth > -1e-6) : [];
    if (wedged.length >= 2) {
      let d = [pos[0] - from[0], pos[1] - from[1]];
      const len = Math.hypot(d[0], d[1]), tol = Math.sin(WEDGE_IN) * len;
      const away = (o) => { const u = [from[0] - o.pos[0], from[1] - o.pos[1]], l = Math.hypot(u[0], u[1]) || 1; return [u[0] / l, u[1] / l]; };
      for (const o of wedged) {
        const n = away(o), k = d[0] * n[0] + d[1] * n[1] + tol;
        if (k < 0) d = [d[0] - k * n[0], d[1] - k * n[1]];
      }
      if (wedged.some((o) => { const n = away(o); return d[0] * n[0] + d[1] * n[1] + tol < -1e-9; })) return from.slice();
      return [from[0] + d[0], from[1] + d[1], pos[2]];
    }
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

  // The point Locomotion heads for on the way to pos, which it should reach facing yaw: on the line through pos along
  // yaw, axisLook ahead of where from projects onto it, but at most axisLead out from pos and never past it.
  axisPoint(pos, yaw, from) {
    const c = Math.cos(yaw), s = Math.sin(yaw);
    const along = c * (from[0] - pos[0]) + s * (from[1] - pos[1]);
    const k = Math.max(-this.p.axisLead, Math.min(0, along + this.p.axisLook));
    return [pos[0] + c * k, pos[1] + s * k];
  }

  // A walk with a facing (axisPoint) is on: the variant has the SceneBot stances and locoTarget is 'stance'.
  axisWalk() { return !!this.stances && this.p.locoTarget !== 'centre'; }

  // {stick, face, arrive} for Locomotion walking to nav ({pos, yaw}, yaw null for no facing). arrive is the pose the
  // query reaches at its last trajectory point (predictTrajectory), or null.
  approach(nav, obstacles) {
    const p = this.p, from = this.sim.pos;
    const togo = Math.hypot(nav.pos[0] - from[0], nav.pos[1] - from[1]);
    if (nav.yaw !== null && p.locoTarget === 'arrive' && togo < p.arriveReach) {
      const face = [Math.cos(nav.yaw), Math.sin(nav.yaw)];
      if (togo <= 0.05) return { stick: null, face, arrive: nav };
      const mag = Math.min(1, togo / (60 * this.dt * p.walkFwd));
      const dir = this.steer(from, [(nav.pos[0] - from[0]) / togo, (nav.pos[1] - from[1]) / togo], togo, obstacles);
      return { stick: [mag * dir[0], mag * dir[1]], face, arrive: nav };
    }
    const aim = nav.yaw === null ? nav.pos : this.axisPoint(nav.pos, nav.yaw, from);
    const d = [aim[0] - from[0], aim[1] - from[1]], dist = Math.hypot(d[0], d[1]);
    if (dist <= 0.05) return { stick: null, face: null, arrive: null };
    // A walk to a box or marker centre slows down as if to stop 0.4 m short of it. The stance is where the robot
    // stands, so a walk to it does not keep that margin.
    const mag = Math.min(1, Math.max(0.35, (nav.yaw === null ? dist - 0.4 : dist) / 1.0));
    const dir = this.steer(from, [d[0] / dist, d[1] / dist], dist, obstacles);
    return { stick: [mag * dir[0], mag * dir[1]], face: null, arrive: null };
  }

  // {pos, yaw} Locomotion walks a pick or a place to: with locoTarget 'stance', the start of the SceneBot pick or place
  // of the side the last search chose for this goal, else the goal's centre with yaw null.
  navTarget(goal, activity) {
    const ns = this.navSide, st = this.axisWalk() ? this.stances[activity] : null;
    if (!st || !st.length || !ns || ns.activity !== activity || ns.id !== (goal.id ?? null)) {
      this.stance = null;
      return { pos: goal.pos, yaw: null };
    }
    const side = targetSides(goal.pos, goal.yaw, this.cfg.box_sides)[ns.side];
    const d = this.dataTarget(side.pos, side.dir, activity), [c, s] = d.dir, dyaw = d.dir === side.dir ? side.yaw : Math.atan2(s, c);
    const at = st.map(([x, y, yaw]) => ({ pos: [d.pos[0] + c * x - s * y, d.pos[1] + s * x + c * y], yaw: dyaw + yaw }));
    const dist = (t) => Math.hypot(t.pos[0] - this.sim.pos[0], t.pos[1] - this.sim.pos[1]);
    const prev = this.stance;
    let k = prev && prev.side === ns.side && prev.activity === activity && prev.k < at.length ? prev.k : -1, best = 0;
    for (let i = 1; i < at.length; i++) if (dist(at[i]) < dist(at[best])) best = i;
    if (k < 0 || dist(at[best]) < dist(at[k]) - this.p.stanceSwitch) k = best;
    this.stance = { side: ns.side, activity, k };
    return at[k];
  }

  // Boxes closer together than the robot can pass between (a chain of gaps under twice the collision radius): per group
  // its centre, the radius around it, the direction across the line between its first two boxes, and its boxes. Tables
  // and the box to pick up (keep) are in no group.
  groupClose(obstacles, keep = null) {
    const r = this.p.collisionRadius, rad = (o) => Math.max(o.half[0], o.half[1]);
    const boxes = obstacles.filter((o) => !o.table && (keep === null || o.id !== keep)), root = boxes.map((_, i) => i);
    const find = (i) => (root[i] === i ? i : (root[i] = find(root[i])));
    for (let i = 0; i < boxes.length; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        const a = boxes[i], b = boxes[j];
        if (Math.hypot(a.pos[0] - b.pos[0], a.pos[1] - b.pos[1]) - rad(a) - rad(b) < 2 * r) root[find(j)] = find(i);
      }
    }
    const groups = new Map();
    boxes.forEach((o, i) => { const k = find(i); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(o); });
    const out = [];
    for (const members of groups.values()) {
      if (members.length < 2) continue;
      const pos = [0, 1].map((k) => members.reduce((a, o) => a + o.pos[k], 0) / members.length);
      const R = Math.max(...members.map((o) => Math.hypot(o.pos[0] - pos[0], o.pos[1] - pos[1]) + rad(o)));
      const [a, b] = members, u = [b.pos[0] - a.pos[0], b.pos[1] - a.pos[1]], l = Math.hypot(u[0], u[1]) || 1;
      out.push({ pos, R, across: [-u[1] / l, u[0] / l], members });
    }
    return out;
  }

  // Turn away from boxes that block the straight line to the goal, to the side they are not on. The boxes of a group
  // (groupClose) all turn to the side their group is not on, so their turns add up and do not lead into a gap. A robot
  // inside a group (it put a box down next to another) first leaves it across the line between the boxes, on its own
  // side, or toward the goal from the middle.
  steer(from, dir, goalDist, obstacles) {
    const groups = this.groupClose(obstacles, this.goalBoxId ?? null), groupOf = new Map();
    for (const g of groups) {
      for (const o of g.members) groupOf.set(o, g);
      const rel = [from[0] - g.pos[0], from[1] - g.pos[1]];
      if (Math.hypot(rel[0], rel[1]) >= g.R) continue;
      const n = g.across, own = n[0] * rel[0] + n[1] * rel[1];
      const s = (Math.abs(own) > GROUP_MIDDLE ? own : n[0] * dir[0] + n[1] * dir[1]) >= 0 ? 1 : -1;
      return [s * n[0], s * n[1]];
    }
    let d = dir.slice();
    for (const o of obstacles) {
      const rel = [o.pos[0] - from[0], o.pos[1] - from[1]];
      const dist = Math.hypot(rel[0], rel[1]);
      const along = rel[0] * dir[0] + rel[1] * dir[1];
      if (dist > 1.6 || along < 0 || along > goalDist) continue;
      const lateral = dir[0] * rel[1] - dir[1] * rel[0];
      const clearance = Math.max(o.half[0], o.half[1]) + this.p.collisionRadius + 0.15;
      if (Math.abs(lateral) >= clearance) continue;
      const g = groupOf.get(o), gl = g ? dir[0] * (g.pos[1] - from[1]) - dir[1] * (g.pos[0] - from[0]) : lateral;
      const side = gl > 0 ? -1 : 1;
      const w = 2 * (1 - Math.abs(lateral) / clearance) * (1 - dist / 1.6);
      d = [d[0] - side * w * dir[1], d[1] + side * w * dir[0]];
    }
    const l = Math.hypot(d[0], d[1]) || 1;
    return [d[0] / l, d[1] / l];
  }

  // face: [x, y] world direction or null. The robot turns there instead of toward the stick, as with the face input of
  // the deploy's box bridge. arrive: {pos} the trajectory reaches at its last point and stops at, or null.
  predictTrajectory(stick, speed, obstacles, face = null, arrive = null) {
    const sim = this.sim, dt = 20 * this.dt;
    const rots = [sim.rot.slice()], drot = [this.desiredRot.slice()];
    const dvel = [this.desiredVel.slice()];
    const faceRot = face ? q.fromYaw(Math.atan2(face[1], face[0])) : null;
    for (let i = 1; i < 4; i++) {
      const dv = this.desiredVelocity(stick, rots[i - 1], speed);
      drot.push(faceRot || (stick ? q.fromYaw(Math.atan2(dv[1], dv[0])) : drot[i - 1]));
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
    if (arrive) {
      // A cubic from the simulation object's position and velocity to rest at arrive. A speed above 3 × distance / time
      // would carry the cubic past arrive and back, so it is cut there.
      const D = [arrive.pos[0] - sim.pos[0], arrive.pos[1] - sim.pos[1], 0];
      const T = 3 * dt, room = 3 * Math.hypot(D[0], D[1]) / T, v = Math.hypot(sim.vel[0], sim.vel[1]);
      const v0 = v > room ? v3.scale([sim.vel[0], sim.vel[1], 0], room / v) : [sim.vel[0], sim.vel[1], 0];
      for (let i = 1; i < 4; i++) {
        const t = i / 3;
        pos[i] = v3.add(sim.pos, v3.add(v3.scale(v0, T * (t - 2 * t * t + t * t * t)), v3.scale(D, 3 * t * t - 2 * t * t * t)));
      }
    }
    // The velocity at the last point: the simulation object's, or none where the trajectory stops at arrive.
    this.trajectory = { pos, rot: rots, vel: arrive ? [0, 0, 0] : s.vel.slice() };
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
    if (qy.length > NPOSE_FEATURES + 14) {
      const tv = q.invMulVec(P.rootRot, this.trajectory.vel);
      qy[NPOSE_FEATURES + 14] = tv[0]; qy[NPOSE_FEATURES + 15] = tv[1];
    }
    return qy;
  }

  boxQuery(targetPos, targetDir, tag) {
    const qy = new Float32Array(this.backend.ctrl.box.nfeatures);
    qy.set(this.poseFeatures());
    const P = this.pose, d = this.dataTarget(targetPos, targetDir, tag);
    let tp = q.invMulVec(P.rootRot, [d.pos[0] - P.rootPos[0], d.pos[1] - P.rootPos[1], 0]);
    const r = Math.hypot(tp[0], tp[1]);
    if (r > this.p.boxQueryRadius) tp = v3.scale(tp, this.p.boxQueryRadius / r);
    const td = q.invMulVec(P.rootRot, [d.dir[0], d.dir[1], 0]);
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

  // The target {pos, dir} of the data's pick or place motion for a target of the world: back by the edit of the motion
  // that plays (boxEdit) and by boxAhead. worldTarget goes the other way. dir: [x, y], unit.
  dataTarget(pos, dir, kind) {
    const [ex, ey, eyaw] = this.boxEdit[kind] ?? NO_EDIT, x = ex + this.p.boxAhead;
    const d = eyaw ? turned(dir, -eyaw) : dir;
    return { pos: [pos[0] - x * d[0] + ey * d[1], pos[1] - x * d[1] - ey * d[0]], dir: d };
  }
  worldTarget(pos, dir, kind) {
    const [ex, ey, eyaw] = this.boxEdit[kind] ?? NO_EDIT, x = ex + this.p.boxAhead;
    return { pos: [pos[0] + x * dir[0] - ey * dir[1], pos[1] + x * dir[1] + ey * dir[0]], dir: eyaw ? turned(dir, eyaw) : dir };
  }

  // Where the playing animation believes the box target is, in world coordinates, as worldTarget moves it.
  animTarget() {
    const b = this.backend;
    if (b.active === 'loco') return null;
    const f = b.denormalize(b.active, b.currentFeatures());
    const P = this.pose;
    const dir = q.mulVec(P.rootRot, [f[NPOSE_FEATURES + 2], f[NPOSE_FEATURES + 3], 0]);
    const at = v3.add(P.rootPos, q.mulVec(P.rootRot, [f[NPOSE_FEATURES], f[NPOSE_FEATURES + 1], 0]));
    const w = this.worldTarget(at, dir, this.boxTag || 'pick');
    return { pos: [w.pos[0], w.pos[1], at[2]], dir: [w.dir[0], w.dir[1], 0] };
  }

  // The held box in world coordinates, from the Decompressor's box outputs, boxAhead farther along its X axis, which
  // points away from the carrier.
  heldBoxWorld() {
    const P = this.pose, ahead = q.mulVec(P.boxRot, [this.p.boxAhead, 0, 0]);
    return { pos: v3.add(P.rootPos, q.mulVec(P.rootRot, v3.add(P.boxPos, ahead))), rot: q.norm(q.mul(P.rootRot, P.boxRot)) };
  }

  // The pelvis height the data has at the playing frame: an edited pick or place keeps the data's moments for bent and
  // upright.
  hipHeight() { return this.pose.hipPos[2] - (this.backend.editLift ? this.backend.editLift() : 0); }
  isLow() { return this.hipHeight() < this.p.zLow; }
  // The clip of the box controller is at or past its last step (fromLastStep). The learned backend has no steps.
  pastLastStep() {
    const b = this.backend;
    return b.kind !== 'mm' || b.active === 'loco' || b.fromLastStep(b.active)[b.frame] === 1;
  }
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

  // input: { stick: [x, y] world direction or null, face: [x, y] world direction or null, walkTo: {pos, yaw} or null
  //          (a click on the floor that Locomotion walks to itself), slow: bool, holding: bool,
  //          pickBox: {pos, yaw, id} or null, placeTarget: {pos, yaw} or null,
  //          boxes: floor boxes [{pos, yaw, id}], boxHalf: [x, y], forceSearch: bool }
  update(input) {
    const dt = this.dt, p = this.p, b = this.backend;
    // The box left the hands while the robot rises with it (the physical grip lost, or the box removed): the tag flips
    // to place, so the robot rises with empty hands, as after a pick cancelled while bent.
    if (this.holding && !input.holding && this.mode === 'box' && this.boxTag === 'pick') this.boxTag = 'place';
    this.holding = input.holding;
    this.events = { grasp: false, release: false };
    const speedScale = input.slow ? p.slowScale : 1;
    const speed = { fwd: p.walkFwd * speedScale, side: p.walkSide * speedScale, back: p.walkBack * speedScale };
    const goal = input.holding ? input.placeTarget : input.pickBox;
    const activity = input.holding ? (input.placeTarget ? 'place' : null) : (input.pickBox ? 'pick' : null);
    this.why = null;
    this.activity = activity;
    this.goalDist = goal ? Math.hypot(goal.pos[0] - this.pose.rootPos[0], goal.pos[1] - this.pose.rootPos[1]) : null;
    const obstacles = input.boxes.filter((o) => !(this.mode === 'box' && input.pickBox && o.id === input.pickBox.id))
      .map((o) => ({ id: o.id, pos: o.pos, yaw: o.yaw, half: input.boxHalf })).concat(input.tables ?? []);

    let stick = input.stick, face = input.face || null, arrive = null;
    let navigating = false;
    this.nav = null;
    const walkTo = goal ? null : input.walkTo || null;
    this.goalBoxId = activity === 'pick' ? input.pickBox.id : null;
    if (!stick && !face && (goal || walkTo) && this.mode === 'loco') {
      const nav = goal ? this.navTarget(goal, activity) : walkTo;
      ({ stick, face, arrive } = this.approach(nav, obstacles));
      if (goal) {
        navigating = !!(stick || arrive);
        this.nav = nav;
      }
    }
    this.navigating = navigating;
    // The arrive state: the query's trajectory ends at a pose (approach), and the search uses the arrive weights.
    this.arriving = !!arrive;
    // The stance of a pick is closer to the box than collisionRadius, so on the way there the box being picked does not
    // stop the simulation object or the trajectory, nor does the table under a pick or a place target. steer still turns
    // around them.
    const toStance = this.nav && this.nav.yaw !== null;
    const blocking = toStance ? obstacles.filter((o) => !(activity === 'pick' && o.id === input.pickBox.id) && !(goal.table && o.id === goal.table)) : obstacles;

    const dvel = this.desiredVelocity(stick, this.sim.rot, speed);
    const dvelChange = v3.len(v3.sub(dvel, this.desiredVel)) / dt;
    this.desiredVel = dvel;
    if (face) this.desiredRot = q.fromYaw(Math.atan2(face[1], face[0]));
    else if (stick && Math.hypot(stick[0], stick[1]) > 0.01) this.desiredRot = q.fromYaw(Math.atan2(dvel[1], dvel[0]));
    this.predictTrajectory(stick, speed, blocking, face, arrive);
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
      const moved = () => Math.hypot(sides[c.side].pos[0] - c.pos[0], sides[c.side].pos[1] - c.pos[1]);
      const turned = () => Math.abs(wrapAngle(sides[c.side].yaw - c.yaw));
      const cause = this.calledOff ? 'check' : activity !== want ? 'cleared' : input.stick ? 'stick'
        : input.pickBox && phase === 'pick-pre' && c.boxId !== input.pickBox.id ? 'other'
          : moved() > p.movedTolerance ? 'moved' : turned() > p.movedAngle ? 'turned' : null;
      abort = cause !== null;
      if (abort && !this.commit) this.commit = c;
      if (abort) {
        this.why = { by: `cancel-${cause}`, ...(cause === 'moved' ? { moved: moved(), limit: p.movedTolerance } : {}),
          ...(cause === 'turned' ? { turned: turned(), limit: p.movedAngle } : {}) };
      }
    }
    this.calledOff = false;
    if (abort) {
      this.boxTag = this.boxTag === 'pick' ? 'place' : 'pick';
      this.forceSearch = true;
    }

    const hold = this.animHold();
    this.holdSwitchTime = this.bentForEvent() && hold && hold[1] > 0.1 && hold[1] < 0.9 ? this.holdSwitchTime + dt : 0;
    const finishing = rising && !this.risen() && !b.endOfAnim();
    this.finishing = finishing;
    const searchDue = !finishing && (this.forceSearch || this.searchTimer <= 0 || (b.kind === 'mm' && b.endOfAnim()));
    if (searchDue) {
      this.searchAt = this.clock;
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
      this.lastEvent = { kind: 'grasp', t: this.clock };
      if (!this.commit && this.target) this.commit = { boxId: input.pickBox.id, side: this.target.side, pos: this.target.pos.slice(), yaw: this.target.yaw, dir: this.target.dir.slice() };
    }
    if (ph === 'place-pre' && held <= p.releaseOff && hb.pos[2] < p.floorBoxZ + (input.placeTarget?.z ?? 0)) {
      this.events.release = true;
      this.lastEvent = { kind: 'release', t: this.clock };
      if (!this.commit && this.target) this.commit = { boxId: null, side: this.target.side, pos: this.target.pos.slice(), yaw: this.target.yaw, dir: this.target.dir.slice() };
    }

    if (this.mode === 'loco') {
      this.simPositionUpdate(this.sim, this.desiredVel, p.simVelocityHalflife, dt, blocking);
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
    this.semantics = this.mode !== 'loco' ? this.boxPhase() : this.arriving ? 'arrive' : navigating ? 'navigate' : 'locomotion';
    this.clock += dt;
    this.trace(!!goal);
  }

  // The page's record of the state machine: each change of state with its cause, which is noted where the decision was
  // made (why), or else told from the two states. A state is the semantics and what plays it: 'navigate:loco',
  // 'navigate:carry', 'pick-pre:pick' (stateOf of states.js).
  trace(hasGoal) {
    const f = this.fsm, s = this.semantics;
    const key = s.startsWith('pick') ? `${s}:pick` : s.startsWith('place') ? `${s}:place` : `${s}:${this.holding ? 'carry' : 'loco'}`;
    if (key === f.state) return;
    if (f.state !== null) {
      f.history.push({ t: this.clock, from: f.state, to: key, ...(this.why || this.inferWhy(f.state, key, hasGoal)) });
      if (f.history.length > FSM_HISTORY) f.history.shift();
    }
    f.state = key;
    f.since = this.clock;
  }

  inferWhy(from, to, hasGoal) {
    const [fs, fk] = from.split(':'), [ts, tk] = to.split(':');
    const walks = (x) => x === 'locomotion' || x === 'navigate' || x === 'arrive';
    const togo = this.nav ? Math.hypot(this.nav.pos[0] - this.sim.pos[0], this.nav.pos[1] - this.sim.pos[1]) : null;
    if (fs === 'pick-pre' && ts === 'pick-post') return { by: 'grasp' };
    if (fs === 'place-pre' && ts === 'place-post') {
      return this.lastEvent && this.lastEvent.kind === 'release' && this.clock - this.lastEvent.t < 0.5 ? { by: 'release' } : { by: 'drop', again: hasGoal };
    }
    if ((fk === 'carry' && tk === 'loco') || (fs === 'pick-post' && ts === 'place-post')) return { by: 'drop', again: hasGoal };
    if (walks(fs) && walks(ts)) {
      if (fk !== tk) return { by: 'held' };
      if (ts === 'arrive') return { by: 'near', togo, reach: this.p.arriveReach };
      if (ts === 'navigate') return fs === 'arrive' ? { by: 'far', togo, reach: this.p.arriveReach } : { by: 'target' };
      return { by: hasGoal ? 'reached' : 'cleared' };
    }
    return { by: 'other' };
  }

  // The cost of the clip that plays, against the query of now: the cost every search starts from, which another frame
  // must beat. The query takes its pose from the playing frame, so the cost is how far the trajectory of the clip
  // (Locomotion) or the box or marker of the clip (box controller) is from the one asked for now. While the robot
  // rises, the target is where the grasp or the release happened.
  playCost() {
    const b = this.backend;
    if (this.fsm.state === null) return null;
    let kind, query;
    if (this.mode === 'loco') {
      kind = this.arriving ? 'arrive' : 'loco';
      query = this.locoQuery(this.holding);
    } else {
      const ph = this.boxPhase(), t = ((ph === 'pick-post' || ph === 'place-post') && this.commit) || this.target;
      if (!t || !this.boxTag) return null;
      kind = this.boxTag;
      query = this.boxQuery(t.pos, t.dir, kind);
    }
    const name = this.dbOf(kind);
    return b.playCost(name, b.normalize(name, query), this.p.gains[kind]);
  }

  // The numbers the rules of the state machine compare now, for the page.
  status() {
    const p = this.p, f = this.fsm, th = this.thresholds(this.activity || this.boxTag || 'pick');
    const cost = {};
    for (const k in QUERY_DB) cost[k] = this.lastMatch[k] ? this.lastMatch[k].cost : null;
    const query = this.mode !== 'loco' ? this.boxTag : this.arriving ? 'arrive' : 'loco', at = this.matchAt[query];
    // Between two searches (every searchTime) the cost is held as usual. It is frozen when the search is skipped
    // (the rise is finished first) or when the last search did not match the query of now.
    const frozen = at === undefined ? null : this.finishing ? 'rise' : at < this.searchAt ? 'unmatched' : null;
    return {
      state: f.state, since: this.clock - f.since, last: f.history.length ? f.history[f.history.length - 1] : null, clock: this.clock,
      mode: this.mode, query, activity: this.activity, cost, frozen, costAge: at === undefined ? null : this.clock - at,
      play: this.playCost(), boxCost: Number.isFinite(this.boxCost) ? this.boxCost : null, sides: this.lastCosts.slice(), side: this.target ? this.target.side : null,
      searchedAt: { ...this.matchAt }, enter: th.enter, leave: th.leave, margin: th.margin,
      dist: this.goalDist, engage: this.activity === 'place' ? p.placeEngageRadius : p.forceEngageRadius, limit: p.takeOverRadius,
      search: p.engageRadius, wait: this.reapproach === null ? null : Math.max(0, p.reapproachTime - this.reapproach),
      togo: this.nav ? Math.hypot(this.nav.pos[0] - this.sim.pos[0], this.nav.pos[1] - this.sim.pos[1]) : null,
      reach: this.axisWalk() && p.locoTarget === 'arrive' ? p.arriveReach : null,
      hip: this.hipHeight(), upright: p.zUpright, low: p.zLow, tilt: this.torsoTilt(), tiltMax: p.uprightTilt,
    };
  }

  // A side is usable when the straight walk to where the robot stands for it (standOff behind the target, and
  // boxAhead more) stays clear of the other floor boxes. The box controller walks straight at the target, only
  // locomotion steers.
  approachClear(from, side, obstacles, kind = 'pick') {
    const p = this.p, t = this.dataTarget(side.pos, side.dir, kind);
    const stand = [t.pos[0] - p.standOff * t.dir[0], t.pos[1] - p.standOff * t.dir[1]];
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
    const m = this.match(this.arriving ? 'arrive' : 'loco', qy);
    this.transition(b.matchPose(m, this.curr.rootPos, this.curr.rootRot, this.dt));
    b.commit(m);
  }

  // The learned backend has one box Projector, so only exact search can use different box databases.
  dbOf(kind) {
    return isBoxQuery(kind) && this.p.boxDb === 'different' && this.backend.db?.[kind] ? kind : QUERY_DB[kind];
  }

  match(kind, query, entry = 0) {
    const b = this.backend, name = this.dbOf(kind), qn = b.normalize(name, query), gain = this.p.gains[kind];
    const sbPre = isBoxQuery(kind) && this.cfg.scenebot_entry_frames && b.sceneBotPre && b.active === name && b.sceneBotPre(name)[b.frame] === 1;
    const mask = isBoxQuery(kind) && this.p.boxEntry === 'walk' && b.noEntry
      ? (sbPre ? b.beforeEventMask(name) : this.walkIn ? b.walkInMask(name) : b.noEntry(name)) : null;
    return { ...b.match(name, qn, gain, undefined, undefined, entry, mask, this.p.transitionCost), kind, qn, gain };
  }

  search(input, sides, activity, goal, forced, obstacles) {
    const b = this.backend, p = this.p, P = this.pose;
    const { enter, leave, margin } = this.thresholds(activity || this.boxTag || 'pick');
    this.lastCosts = [];
    this.navSide = null;
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
        const others = obstacles.filter((o) => (!input.pickBox || o.id !== input.pickBox.id) && o.id !== goal.table);
        // A target on a table is taken from outside the table only (World.sidesFor).
        const open = goal.sides ? sides.filter((s) => goal.sides.includes(s.side)) : sides;
        let cands = locked ? [sides[this.target.side]] : open.filter((s) => this.approachClear(P.rootPos, s, others, activity));
        // A target pressed against another box blocks every side, so the robot takes any side once it is there.
        if (!cands.length && near < p.approachFallback) cands = open;
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
          this.navSide = { side: best.s.side, activity, id: goal.id ?? null };
        }
      }
    }
    this.boxCost = boxMatch ? boxMatch.cost : Infinity;
    if (boxMatch) { this.lastMatch[boxMatch.kind] = boxMatch; this.matchAt[boxMatch.kind] = this.clock; }

    let useBox = false;
    const engage = activity === 'place' ? p.placeEngageRadius : p.forceEngageRadius;
    if (this.leaveBox || this.missed) {
      this.why = !this.leaveBox ? { by: 'leave-missed' }
        : { by: 'leave-moved', moved: Math.hypot(goal.pos[0] - this.engaged.pos[0], goal.pos[1] - this.engaged.pos[1]), limit: p.leaveMoved };
      this.reapproach = 0;
      this.engaged = null;
      this.walkIn = true;
    } else if (this.mode === 'box') {
      if (phase === 'pick-post' || phase === 'place-post') {
        useBox = !!boxMatch;
        if (!useBox) this.why = { by: 'upright', hip: this.hipHeight(), tilt: this.torsoTilt() };
      } else if (!activity || input.stick) {
        useBox = !this.isUpright() && !!boxMatch;
        if (!useBox) this.why = { by: !activity ? 'leave-cleared' : 'leave-stick' };
      } else {
        // Near the target the robot stays in the box controller even at a high cost, as it entered it.
        const dist = Math.hypot(goal.pos[0] - P.rootPos[0], goal.pos[1] - P.rootPos[1]);
        useBox = !!boxMatch && !(this.isUpright() && this.boxCost > leave && dist > engage + 0.3);
        if (!useBox) this.why = boxMatch ? { by: 'leave-cost', cost: this.boxCost, leave, dist, radius: engage + 0.3 } : { by: 'leave-nomatch' };
      }
    } else if (boxMatch && !input.stick) {
      const dist = Math.hypot(goal.pos[0] - P.rootPos[0], goal.pos[1] - P.rootPos[1]);
      const within = p.takeOverRadius === null || dist < p.takeOverRadius;
      const byCost = this.boxCost < enter, byDist = dist < engage && this.reapproach === null;
      useBox = within && (byCost || byDist);
      if (useBox) {
        this.boxTag = activity;
        this.reapproach = null;
        this.why = { by: byCost && byDist ? 'both' : byCost ? 'cost' : 'distance', cost: this.boxCost, enter, dist, radius: engage,
          limit: p.takeOverRadius, side: this.target ? this.target.side : null };
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
      const kind = this.arriving ? 'arrive' : 'loco';
      m = this.match(kind, this.locoQuery(input.holding));
      this.lastMatch[kind] = m;
      this.matchAt[kind] = this.clock;
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
