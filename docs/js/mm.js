// Real-time motion matching in the browser -- a 1:1 port of mm_g1/controller.py
// (GenoView smoothed-sim-root locomotion + the J jump skill + the B pick/carry/place box
// skill). Reads the per-skill databases + box arrays exported by tools/export_web_data.py.
//
// The box state machine mirrors the Python controller exactly:
//   LOCOMOTION --B (near box)--> PICK (ride) --> CARRY (search) --B--> PLACE (ride) --> LOCO
// PICK/PLACE are ridden (entered by a nearest-neighbour match of the live pose + box pose,
// then played to the phase end); CARRY is searched like locomotion but among CARRY frames
// with the box pose added to the query. The box rides the controller root while held.

import { quat, v3 } from './quat.js';

const FORWARD = [1, 0, 0];
const IDENTITY = [1, 0, 0, 0];

// ---- DB loader: typed-array views into mm.bin per the mm.json header ----
export function loadDB(meta, buf) {
  const A = {};
  for (const [name, h] of Object.entries(meta.arrays)) {
    const TA = h.dtype === 'int32' ? Int32Array : Float32Array;
    const count = h.shape.reduce((a, b) => a * b, 1);
    A[name] = count ? new TA(buf, h.offset, count) : new TA(0);
  }
  return A;
}

// ---- spring + inertialization helpers (port of mm_g1/springs.py) ----
const damp = (hl) => (4.0 * 0.69314718056) / (hl + 1e-5);

function decayPos(x, v, hl, dt) {
  const y = damp(hl) / 2, e = Math.exp(-y * dt), xo = [], vo = [];
  for (let i = 0; i < x.length; i++) {
    const j1 = v[i] + x[i] * y;
    xo[i] = e * (x[i] + j1 * dt);
    vo[i] = e * (v[i] - j1 * y * dt);
  }
  return [xo, vo];
}
function decayRot(x, v, hl, dt) {
  const y = damp(hl) / 2, e = Math.exp(-y * dt);
  const j0 = quat.toScaledAngleAxis(x);
  const j1 = [v[0] + j0[0] * y, v[1] + j0[1] * y, v[2] + j0[2] * y];
  const q = quat.fromScaledAngleAxis([e * (j0[0] + j1[0] * dt), e * (j0[1] + j1[1] * dt), e * (j0[2] + j1[2] * dt)]);
  return [q, [e * (v[0] - j1[0] * y * dt), e * (v[1] - j1[1] * y * dt), e * (v[2] - j1[2] * y * dt)]];
}
function trajPos(pos, vel, acc, dvel, hl, dt) {
  const y = damp(hl) / 2, e = Math.exp(-y * dt), P = [], V = [], Ac = [];
  for (let i = 0; i < 3; i++) {
    const j0 = vel[i] - dvel[i], j1 = acc[i] + j0 * y;
    P[i] = e * ((-j1) / (y * y) + (-j0 - j1 * dt) / y) + j1 / (y * y) + j0 / y + dvel[i] * dt + pos[i];
    V[i] = e * (j0 + j1 * dt) + dvel[i];
    Ac[i] = e * (acc[i] - j1 * y * dt);
  }
  return [P, V, Ac];
}
function trajRot(rot, ang, dRot, hl, dt) {
  const y = damp(hl) / 2, e = Math.exp(-y * dt);
  const j0 = quat.toScaledAngleAxis(quat.abs(quat.mul_inv(rot, dRot)));
  const j1 = [ang[0] + j0[0] * y, ang[1] + j0[1] * y, ang[2] + j0[2] * y];
  const q = quat.mul(quat.fromScaledAngleAxis([e * (j0[0] + j1[0] * dt), e * (j0[1] + j1[1] * dt), e * (j0[2] + j1[2] * dt)]), dRot);
  return [q, [e * (ang[0] - j1[0] * y * dt), e * (ang[1] - j1[1] * y * dt), e * (ang[2] - j1[2] * y * dt)]];
}

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

export class MotionMatcher {
  constructor(meta, A) {
    this.A = A;
    this.fps = meta.fps; this.DT = 1 / meta.fps;
    this.H = Math.max(...meta.horizons);
    this.Ttimes = meta.horizons.map((h) => h / meta.fps);
    this.MAX_SPEED = meta.max_speed; this.WALK_SCALE = meta.walk_scale;
    this.CARRY_MAX_SPEED = meta.carry_max_speed;
    this.SEARCH_TIME = meta.search_time; this.CURRENT_BIAS = meta.current_bias;
    this.INERT = meta.inert_halflife; this.VEL_HL = meta.vel_halflife; this.ROT_HL = meta.rot_halflife;
    this.BOX_INERT = meta.box_inert_halflife;
    this.PHASE_TD = meta.phase_touchdown; this.PHASE_AFTER = meta.phase_after;
    this.PICK_RADIUS = meta.pick_radius;
    this.BOX_SPAWN_FWD = meta.box_spawn_fwd; this.BOX_SPAWN_LAT = meta.box_spawn_lat;
    this.BOX_REST_Z = meta.box_rest_z;
    this.LOCO = meta.skill_loco; this.PICK = meta.skill_pick;
    this.CARRY = meta.skill_carry; this.PLACE = meta.skill_place;
    this.clipNames = meta.clip_names;

    this.starts = A.starts; this.stops = A.stops;
    this.Xloco = A.Xloco; this.Xcarry = A.Xcarry;
    this.off = { loco: A.locoOffset, carry: A.carryOffset, pick: A.pickOffset, place: A.placeOffset };
    this.scl = { loco: A.locoScale, carry: A.carryScale, pick: A.pickScale, place: A.placeScale };
    this.boxSpawnRot = Array.from(A.box_spawn_rot);

    // Loco search clips (skill all 0) and contiguous CARRY segments, as [start, stop] spans.
    this.locoSegs = Array.from(A.search_clips, (ci) => [A.starts[ci], A.stops[ci]]);
    this.carrySegs = [];
    for (let i = 0; i < A.carry_segs.length; i += 2) this.carrySegs.push([A.carry_segs[i], A.carry_segs[i + 1]]);

    // Pick / place entry frames (+ their ride-end frame and pre-built db rows).
    this.pickEnter = A.pick_enter; this.pickEnd = A.pick_end; this.pickEnterX = A.pickEnterX;
    this.placeEnter = A.place_enter; this.placeEnd = A.place_end; this.placeEnterX = A.placeEnterX;

    this.jumpEnter = A.jump_enter; this.jumpLand = A.jump_land;  // parallel arrays (legacy J)
    this.reset();
  }

  // row accessors (return plain arrays)
  _row(arr, d, f) { const b = f * d, r = new Array(d); for (let i = 0; i < d; i++) r[i] = arr[b + i]; return r; }
  dof(f) { return this._row(this.A.dof, 29, f); }
  dofVel(f) { return this._row(this.A.dofVel, 29, f); }
  simPos(f) { return this._row(this.A.simPos, 3, f); }
  simVel(f) { return this._row(this.A.simVel, 3, f); }
  simTheta(f) { return this.A.simTheta[f]; }
  yawRate(f) { return this.A.yawRate[f]; }
  pelvLocalPos(f) { return this._row(this.A.pelvLocalPos, 3, f); }
  pelvLocalVel(f) { return this._row(this.A.pelvLocalVel, 3, f); }
  pelvLocalRot(f) { return this._row(this.A.pelvLocalRot, 4, f); }
  pelvLocalAng(f) { return this._row(this.A.pelvLocalAng, 3, f); }
  rawXpos(f) { return this._row(this.A.rawXpos, 6, f); }
  rawXvel(f) { return this._row(this.A.rawXvel, 9, f); }
  boxLocalPos(f) { return this._row(this.A.boxLocalPos, 3, f); }
  boxLocalRot(f) { return this._row(this.A.boxLocalRot, 4, f); }
  boxLocalPosVel(f) { return this._row(this.A.boxLocalPosVel, 3, f); }
  boxLocalAng(f) { return this._row(this.A.boxLocalAng, 3, f); }
  attach(f) { return this.A.box_attach[f] !== 0; }

  _clipOf(f) { let r = 0; for (let i = 0; i < this.starts.length; i++) { if (this.starts[i] <= f) r = i; else break; } return r; }
  _clipBounds(f) { const r = this._clipOf(f); return [this.starts[r], this.stops[r]]; }

  reset() {
    const sf = Math.min(this.stops[0] - 1, this.starts[0] + 30);
    [this.lo, this.hi] = this._clipBounds(sf);
    this.animFrame = sf;
    this.state = this.LOCO;
    this.rootPos = this.simPos(sf);
    this.rootVel = [0, 0, 0]; this.rootAcc = [0, 0, 0]; this.rootAng = [0, 0, 0];
    this.rootYaw = this.simTheta(sf);
    this.rootRot = quat.yaw(this.rootYaw);
    this.desiredDir = quat.mulVec(this.rootRot, FORWARD);
    this.offDof = new Array(29).fill(0); this.offDofVel = new Array(29).fill(0);
    this.offPP = [0, 0, 0]; this.offPPVel = [0, 0, 0];
    this.offPR = IDENTITY.slice(); this.offPAng = [0, 0, 0];
    this.searchTimer = 0;
    this.jumpPending = false; this.jumpLocked = 0;
    this.boxPending = false; this.boxLocked = 0;
    // Box world pose: a reachable distance in front of the robot's start facing, resting
    // height, oriented as in the data (box_spawn_rot is base-local -> compose with the root).
    this.boxPos = v3.add(this.rootPos,
      quat.mulVec(this.rootRot, [this.BOX_SPAWN_FWD, this.BOX_SPAWN_LAT, 0]));
    this.boxPos[2] = this.BOX_REST_Z;
    this.boxRot = quat.mul(this.rootRot, this.boxSpawnRot);
    this.boxVelWorld = [0, 0, 0]; this.boxPosPrev = this.boxPos.slice();
    this.boxHeld = false;
    this.offBoxP = [0, 0, 0]; this.offBoxPVel = [0, 0, 0];
    this.offBoxR = IDENTITY.slice(); this.offBoxAng = [0, 0, 0];
    this.Tpos = [this.rootPos, this.rootPos, this.rootPos];
    this.Tdir = [this.desiredDir, this.desiredDir, this.desiredDir];
  }

  // ---- public state / triggers ----
  triggerJump() { if (this.jumpLocked === 0 && this.boxLocked === 0 && this.state === this.LOCO) this.jumpPending = true; }
  triggerBox() { if (this.boxLocked === 0 && this.jumpLocked === 0) this.boxPending = true; }
  get jumping() { return this.jumpLocked > 0; }
  get cur() { return this.animFrame; }
  get boxHeldNow() { return this.boxHeld; }
  get nearBox() { return Math.hypot(this.rootPos[0] - this.boxPos[0], this.rootPos[1] - this.boxPos[1]) < this.PICK_RADIUS; }
  stateName() { return ({ [this.LOCO]: 'LOCOMOTION', [this.PICK]: 'PICK', [this.CARRY]: 'CARRY', [this.PLACE]: 'PLACE' })[this.state]; }
  clipName(f) { return this.clipNames[this._clipOf(f)]; }
  boxQpos() { return [this.boxPos[0], this.boxPos[1], this.boxPos[2], this.boxRot[0], this.boxRot[1], this.boxRot[2], this.boxRot[3]]; }

  // ---- jump skill (legacy J) ----
  _bestJumpEntry() {
    if (this.jumpEnter.length === 0) return -1;
    let bi = -1, bd = Infinity; const ba = this.animFrame * 27, X = this.Xloco;
    for (let e = 0; e < this.jumpEnter.length; e++) {
      const base = this.jumpEnter[e] * 27; let s = 0;
      for (let i = 0; i < 27; i++) { const d = X[base + i] - X[ba + i]; s += d * d; }
      if (s < bd) { bd = s; bi = this.jumpEnter[e]; this._bestJumpIdx = e; }
    }
    return bi;
  }

  // ---- inertialized cut (captures the pose + held-box discontinuity) ----
  _inertInto(b, lo, hi) {
    const a = this.animFrame;
    const da = this.dof(a), db = this.dof(b), va = this.dofVel(a), vb = this.dofVel(b);
    for (let i = 0; i < 29; i++) { this.offDof[i] += da[i] - db[i]; this.offDofVel[i] += va[i] - vb[i]; }
    const pa = this.pelvLocalPos(a), pb = this.pelvLocalPos(b), qa = this.pelvLocalVel(a), qb = this.pelvLocalVel(b);
    for (let i = 0; i < 3; i++) { this.offPP[i] += pa[i] - pb[i]; this.offPPVel[i] += qa[i] - qb[i]; }
    this.offPR = quat.abs(quat.mul_inv(quat.mul(this.offPR, this.pelvLocalRot(a)), this.pelvLocalRot(b)));
    const aa = this.pelvLocalAng(a), ab = this.pelvLocalAng(b);
    for (let i = 0; i < 3; i++) this.offPAng[i] += aa[i] - ab[i];
    if (this.boxHeld) {
      const bpa = this.boxLocalPos(a), bpb = this.boxLocalPos(b), bva = this.boxLocalPosVel(a), bvb = this.boxLocalPosVel(b);
      for (let i = 0; i < 3; i++) { this.offBoxP[i] += bpa[i] - bpb[i]; this.offBoxPVel[i] += bva[i] - bvb[i]; }
      this.offBoxR = quat.abs(quat.mul_inv(quat.mul(this.offBoxR, this.boxLocalRot(a)), this.boxLocalRot(b)));
      const baa = this.boxLocalAng(a), bab = this.boxLocalAng(b);
      for (let i = 0; i < 3; i++) this.offBoxAng[i] += baa[i] - bab[i];
    }
    this.animFrame = b; this.lo = lo; this.hi = hi;
  }

  // ---- query assembly (one normalized vector per database, build_db block order) ----
  _query(name) {
    const qh = quat.yaw(this.rootYaw);
    const q = this.rawXpos(this.animFrame).concat(this.rawXvel(this.animFrame));   // pose (15)
    if (name === 'loco' || name === 'carry') {
      for (let k = 0; k < 3; k++) {                                                 // traj pos (6)
        const dp = quat.invMulVec(qh, v3.sub(this.Tpos[k], this.rootPos));
        q.push(dp[0], dp[1]);
      }
      for (let k = 0; k < 3; k++) {                                                 // traj dir (6)
        const dd = quat.invMulVec(qh, this.Tdir[k]);
        q.push(dd[0], dd[1]);
      }
    }
    if (name === 'carry' || name === 'pick' || name === 'place') {                  // box (9)
      const bp = quat.invMulVec(qh, v3.sub(this.boxPos, this.rootPos));
      const baa = quat.toScaledAngleAxis(quat.abs(quat.mul(quat.inv(qh), this.boxRot)));
      const bv = quat.invMulVec(qh, this.boxVelWorld);
      q.push(bp[0], bp[1], bp[2], baa[0], baa[1], baa[2], bv[0], bv[1], bv[2]);
    }
    const off = this.off[name], scl = this.scl[name];
    for (let i = 0; i < q.length; i++) q[i] = (q[i] - off[i]) / scl[i];
    return q;
  }

  // ---- generic per-segment brute-force search (KD-tree equivalent) ----
  // segs: [[rs,re],...]; Xmat/dim: candidate database; Xq: normalized query. withBias adds the
  // genoview stay-in-place bias measured in the SAME feature space. Returns [frame, lo, hi].
  _search(segs, Xq, Xmat, dim, withBias) {
    let bestF = this.animFrame, bestLo = this.lo, bestHi = this.hi;
    let best;
    if (withBias && this.animFrame < this.hi - this.H) {
      let s = 0; const b = this.animFrame * dim;
      for (let i = 0; i < dim; i++) { const d = Xq[i] - Xmat[b + i]; s += d * d; }
      best = Math.sqrt(s) - this.CURRENT_BIAS;
    } else best = Infinity;
    if (best <= 0) return [bestF, bestLo, bestHi];      // current frame already wins
    let bestSq = best * best;
    for (const [rs, re] of segs) {
      const lim = re - this.H;
      for (let f = rs; f < lim; f++) {
        const b = f * dim; let s = 0;
        for (let i = 0; i < dim; i++) { const d = Xq[i] - Xmat[b + i]; s += d * d; if (s >= bestSq) { s = -1; break; } }
        if (s >= 0) { bestSq = s; bestF = f; bestLo = rs; bestHi = re; }
      }
    }
    return [bestF, bestLo, bestHi];
  }

  _searchLoco() {
    const [f, lo, hi] = this._search(this.locoSegs, this._query('loco'), this.Xloco, 27, true);
    if (f !== this.animFrame) this._inertInto(f, lo, hi); else { this.lo = lo; this.hi = hi; }
  }
  _searchCarry() {
    const [f, lo, hi] = this._search(this.carrySegs, this._query('carry'), this.Xcarry, 36, true);
    if (f !== this.animFrame) this._inertInto(f, lo, hi); else { this.lo = lo; this.hi = hi; }
  }
  _bestCarry() {
    if (this.carrySegs.length === 0) return null;
    return this._search(this.carrySegs, this._query('carry'), this.Xcarry, 36, false);
  }
  _bestLoco() { return this._search(this.locoSegs, this._query('loco'), this.Xloco, 27, false); }

  // ---- box skill: enter a pick/place ride, finish it ----
  _enterSkill(enter, end, enterX, name, skill) {
    if (enter.length === 0) return;
    const Xq = this._query(name), d = 24;
    let bi = 0, bd = Infinity;
    for (let r = 0; r < enter.length; r++) {
      let s = 0; const b = r * d;
      for (let i = 0; i < d; i++) { const e = enterX[b + i] - Xq[i]; s += e * e; }
      if (s < bd) { bd = s; bi = r; }
    }
    const entry = enter[bi], phaseEnd = end[bi];
    const lo = this._clipBounds(entry)[0];
    this._inertInto(entry, lo, phaseEnd + 1);            // hi caps the ride at the phase end
    this.state = skill;
    this.boxLocked = Math.max(1, phaseEnd - entry);
    this.searchTimer = this.SEARCH_TIME;
  }

  _maybeTriggerBox() {
    if (!this.boxPending || this.boxLocked > 0 || this.jumpLocked > 0) { this.boxPending = false; return; }
    this.boxPending = false;
    if (this.state === this.LOCO && this.nearBox) {
      this._enterSkill(this.pickEnter, this.pickEnd, this.pickEnterX, 'pick', this.PICK);
    } else if (this.state === this.CARRY) {
      this._enterSkill(this.placeEnter, this.placeEnd, this.placeEnterX, 'place', this.PLACE);
    }
  }

  _finishRide() {
    if (this.state === this.PICK) {                      // pick -> carry
      const res = this._bestCarry();
      if (res) { this._inertInto(res[0], res[1], res[2]); this.state = this.CARRY; }
    } else {                                             // place -> locomotion
      const [f, lo, hi] = this._bestLoco();
      this._inertInto(f, lo, hi); this.state = this.LOCO;
    }
    this.searchTimer = this.SEARCH_TIME;
  }

  // ---- predict the desired trajectory (command springs) ----
  _predictTrajectory(desiredVel, desiredFace) {
    if (v3.norm(desiredFace) > 0.01) this.desiredDir = v3.scale(desiredFace, 1 / v3.norm(desiredFace));
    else if (v3.norm(desiredVel) > 0.01) this.desiredDir = v3.scale(desiredVel, 1 / v3.norm(desiredVel));
    const desiredRot = quat.yaw(Math.atan2(this.desiredDir[1], this.desiredDir[0]));
    this.Tpos = []; this.Tdir = [];
    for (let k = 0; k < 3; k++) {
      const [P] = trajPos(this.rootPos, this.rootVel, this.rootAcc, desiredVel, this.VEL_HL, this.Ttimes[k]);
      this.Tpos.push(P);
      const [Q] = trajRot(this.rootRot, this.rootAng, desiredRot, this.ROT_HL, this.Ttimes[k]);
      this.Tdir.push(quat.mulVec(Q, FORWARD));
    }
  }

  _maybeEnterJump() {
    if (this.jumpPending && this.jumpLocked === 0 && this.boxLocked === 0) {
      this.jumpPending = false;
      const entry = this._bestJumpEntry();
      if (entry >= 0) {
        const [lo, hi] = this._clipBounds(entry);
        this._inertInto(entry, lo, hi);
        const land = this.jumpLand[this._bestJumpIdx];
        const afterEnd = Math.min(land + 1 + this.PHASE_TD + this.PHASE_AFTER, hi - 1);
        this.jumpLocked = Math.max(1, afterEnd - entry);
        this.searchTimer = this.SEARCH_TIME;
      }
    }
  }

  // ---- box reconstruction (rides the root while held, frozen otherwise) ----
  _updateBox(f) {
    const attached = (this.state === this.PICK || this.state === this.CARRY || this.state === this.PLACE) && this.attach(f);
    if (attached) {
      if (!this.boxHeld) {                               // grab: seed the base-local offset
        const localP = quat.invMulVec(this.rootRot, v3.sub(this.boxPos, this.rootPos));
        const localR = quat.mul(quat.inv(this.rootRot), this.boxRot);
        const blp = this.boxLocalPos(f);
        this.offBoxP = [localP[0] - blp[0], localP[1] - blp[1], localP[2] - blp[2]];
        this.offBoxPVel = [0, 0, 0];
        this.offBoxR = quat.abs(quat.mul_inv(localR, this.boxLocalRot(f)));
        this.offBoxAng = [0, 0, 0];
        this.boxHeld = true;
      }
      [this.offBoxP, this.offBoxPVel] = decayPos(this.offBoxP, this.offBoxPVel, this.BOX_INERT, this.DT);
      [this.offBoxR, this.offBoxAng] = decayRot(this.offBoxR, this.offBoxAng, this.BOX_INERT, this.DT);
      const blp = v3.add(this.boxLocalPos(f), this.offBoxP);
      const blr = quat.mul(this.offBoxR, this.boxLocalRot(f));
      this.boxPos = v3.add(this.rootPos, quat.mulVec(this.rootRot, blp));
      this.boxRot = quat.mul(this.rootRot, blr);
    } else {
      this.boxHeld = false;                              // frozen at its current world pose
    }
    this.boxVelWorld = v3.scale(v3.sub(this.boxPos, this.boxPosPrev), 1 / this.DT);
    this.boxPosPrev = this.boxPos.slice();
  }

  // ---- one real-time frame ----
  // desiredVel: [x,y,0] m/s (WASD). desiredFace: [x,y,0] unit or [0,0,0] (arrows).
  step(desiredVel, desiredFace) {
    this._predictTrajectory(desiredVel, desiredFace);
    this._maybeEnterJump();
    this._maybeTriggerBox();

    // ---- Search (skipped while riding a jump or a pick/place skill) ----
    if (this.jumpLocked === 0 && this.boxLocked === 0 && this.searchTimer <= 0) {
      if (this.state === this.CARRY) this._searchCarry(); else this._searchLoco();
      this.searchTimer = this.SEARCH_TIME;
    }

    // ---- Advance the playhead within its current bounds ----
    this.animFrame = clamp(this.animFrame + 1, this.lo, this.hi - 1);
    this.searchTimer -= this.DT;
    if (this.jumpLocked > 0) { this.jumpLocked -= 1; if (this.jumpLocked === 0) this.searchTimer = 0; }
    else if (this.boxLocked > 0) { this.boxLocked -= 1; if (this.boxLocked === 0) this._finishRide(); }
    else if (this.animFrame >= this.hi - 2) this.searchTimer = 0;
    const f = this.animFrame;

    // ---- Integrate controller root from the matched clip's smooth root velocity ----
    const [, , acc] = trajPos(this.rootPos, this.rootVel, this.rootAcc, desiredVel, this.ROT_HL, this.DT);
    this.rootAcc = acc;
    const qhClip = quat.yaw(this.simTheta(f));
    const clipVelLocal = quat.invMulVec(qhClip, this.simVel(f));
    this.rootVel = quat.mulVec(this.rootRot, clipVelLocal);
    this.rootAng = [0, 0, this.yawRate(f)];
    this.rootPos = v3.add(this.rootPos, v3.scale(this.rootVel, this.DT));
    this.rootYaw += this.yawRate(f) * this.DT;
    this.rootRot = quat.yaw(this.rootYaw);

    // ---- Inertialize joints + pelvis-local offset, reconstruct pose ----
    [this.offDof, this.offDofVel] = decayPos(this.offDof, this.offDofVel, this.INERT, this.DT);
    [this.offPP, this.offPPVel] = decayPos(this.offPP, this.offPPVel, this.INERT, this.DT);
    [this.offPR, this.offPAng] = decayRot(this.offPR, this.offPAng, this.INERT, this.DT);

    const dof = this.dof(f), dofOut = new Array(29);
    for (let i = 0; i < 29; i++) dofOut[i] = dof[i] + this.offDof[i];
    const plp = v3.add(this.pelvLocalPos(f), this.offPP);
    const plr = quat.mul(this.offPR, this.pelvLocalRot(f));
    const pelvPos = v3.add(this.rootPos, quat.mulVec(this.rootRot, plp));
    const pelvRot = quat.mul(this.rootRot, plr);

    // ---- Reconstruct the box (rides the root while held; frozen otherwise) ----
    this._updateBox(f);

    const qpos = new Float64Array(36);
    qpos[0] = pelvPos[0]; qpos[1] = pelvPos[1]; qpos[2] = pelvPos[2];
    qpos[3] = pelvRot[0]; qpos[4] = pelvRot[1]; qpos[5] = pelvRot[2]; qpos[6] = pelvRot[3];
    for (let i = 0; i < 29; i++) qpos[7 + i] = dofOut[i];
    return qpos;
  }
}
