// Vectors are [x, y, z] arrays, quaternions [w, x, y, z]. The world is Z-up, +X forward.
// Springs and inertializers are ports of spring.h from orangeduck/Motion-Matching.

export const LN2 = Math.LN2;

export const v3 = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  len: (a) => Math.hypot(a[0], a[1], a[2]),
  lerp: (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t],
  norm: (a, eps = 1e-8) => { const l = Math.hypot(a[0], a[1], a[2]) + eps; return [a[0] / l, a[1] / l, a[2] / l]; },
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
};

export const q = {
  id: () => [1, 0, 0, 0],
  mul: (a, b) => [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]],
  inv: (a) => [a[0], -a[1], -a[2], -a[3]],
  norm: (a) => { const l = Math.hypot(a[0], a[1], a[2], a[3]); return [a[0] / l, a[1] / l, a[2] / l, a[3] / l]; },
  abs: (a) => (a[0] < 0 ? [-a[0], -a[1], -a[2], -a[3]] : a),
  mulVec(a, v) {
    const [w, x, y, z] = a;
    const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
    return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
  },
  invMulVec: (a, v) => q.mulVec(q.inv(a), v),
  mulInv: (a, b) => q.mul(a, q.inv(b)),
  invMul: (a, b) => q.mul(q.inv(a), b),
  fromAngleAxis(angle, axis) {
    const s = Math.sin(angle / 2);
    return [Math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s];
  },
  fromYaw: (yaw) => [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)],
  yaw(a) {
    const f = q.mulVec(a, [1, 0, 0]);
    return Math.atan2(f[1], f[0]);
  },
  toScaledAngleAxis(a, eps = 1e-8) {
    const l = Math.hypot(a[1], a[2], a[3]);
    const half = Math.atan2(l, a[0]);
    const s = (2 * half) / (l + eps);
    return [a[1] * s, a[2] * s, a[3] * s];
  },
  fromScaledAngleAxis(v, eps = 1e-8) {
    const angle = Math.hypot(v[0], v[1], v[2]);
    if (angle < eps) return [1, v[0] / 2, v[1] / 2, v[2] / 2];
    return q.fromAngleAxis(angle, [v[0] / angle, v[1] / angle, v[2] / angle]);
  },
  slerpShortestApprox(a, b, t) {
    const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    const bb = d < 0 ? [-b[0], -b[1], -b[2], -b[3]] : b;
    return q.norm([a[0] + (bb[0] - a[0]) * t, a[1] + (bb[1] - a[1]) * t, a[2] + (bb[2] - a[2]) * t, a[3] + (bb[3] - a[3]) * t]);
  },
  nlerpShortest: (a, b, t) => q.slerpShortestApprox(a, b, t),
  angleBetween(a, b) {
    const d = q.abs(q.mulInv(a, b));
    return 2 * Math.acos(Math.min(1, Math.abs(d[0])));
  },
  // Two matrix columns [c0x, c1x, c0y, c1y, c0z, c1z] (row-major 3x2) -> quaternion
  fromXformXY(t) {
    let c0 = [t[0], t[2], t[4]], c1 = [t[1], t[3], t[5]];
    let c2 = v3.norm(v3.cross(c0, c1));
    c1 = v3.norm(v3.cross(c2, c0));
    c0 = v3.norm(c0);
    return q.fromMatrix(c0, c1, c2);
  },
  fromMatrix(c0, c1, c2) {
    const m00 = c0[0], m10 = c0[1], m20 = c0[2], m01 = c1[0], m11 = c1[1], m21 = c1[2], m02 = c2[0], m12 = c2[1], m22 = c2[2];
    let r;
    if (m22 < 0) {
      r = m00 > m11 ? [m21 - m12, 1 + m00 - m11 - m22, m10 + m01, m02 + m20]
                    : [m02 - m20, m10 + m01, 1 - m00 + m11 - m22, m21 + m12];
    } else {
      r = m00 < -m11 ? [m10 - m01, m02 + m20, m21 + m12, 1 - m00 - m11 + m22]
                     : [1 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01];
    }
    return q.norm(r);
  },
};

export const halflifeToDamping = (h, eps = 1e-5) => (4 * LN2) / (h + eps);
export const dampFactor = (h, dt, eps = 1e-5) => 1 - Math.exp(-(LN2 * dt) / (h + eps));

export function simpleSpringDamperQuat(state, goal, halflife, dt) {
  const y = halflifeToDamping(halflife) / 2;
  const j0 = q.toScaledAngleAxis(q.abs(q.mul(state.x, q.inv(goal))));
  const j1 = v3.add(state.v, v3.scale(j0, y));
  const e = Math.exp(-y * dt);
  state.x = q.mul(q.fromScaledAngleAxis(v3.scale(v3.add(j0, v3.scale(j1, dt)), e)), goal);
  state.v = v3.scale(v3.sub(state.v, v3.scale(j1, y * dt)), e);
}

export function decaySpringVec(x, v, halflife, dt) {
  const y = halflifeToDamping(halflife) / 2;
  const j1 = v3.add(v, v3.scale(x, y));
  const e = Math.exp(-y * dt);
  return [v3.scale(v3.add(x, v3.scale(j1, dt)), e), v3.scale(v3.sub(v, v3.scale(j1, y * dt)), e)];
}

export function decaySpringQuat(x, v, halflife, dt) {
  const y = halflifeToDamping(halflife) / 2;
  const j0 = q.toScaledAngleAxis(x);
  const j1 = v3.add(v, v3.scale(j0, y));
  const e = Math.exp(-y * dt);
  return [q.fromScaledAngleAxis(v3.scale(v3.add(j0, v3.scale(j1, dt)), e)), v3.scale(v3.sub(v, v3.scale(j1, y * dt)), e)];
}

export function decaySpringScalar(x, v, halflife, dt) {
  const y = halflifeToDamping(halflife) / 2;
  const j1 = v + x * y;
  const e = Math.exp(-y * dt);
  return [e * (x + j1 * dt), e * (v - j1 * y * dt)];
}

export function wrapAngle(a) {
  return Math.atan2(Math.sin(a), Math.cos(a));
}
