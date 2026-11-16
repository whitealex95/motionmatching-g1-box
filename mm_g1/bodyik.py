"""Port of lmm-g1-box's web/js/edit/bodyik.js: the whole-body inverse kinematics of the G1 on the MuJoCo model of
the page's physical mode (data/physics/scene.xml). One Gauss-Newton step solves

    (J^T W J + lambda I) dq = J^T W e

over the 35 velocities of the robot (floating pelvis 6, joints 29), with the Jacobians from MuJoCo. The tasks are
the pose of both feet, the pose of both palm sites, the horizontal centre of mass and the joint angles of the pose
the solve starts from. A pose key adds the pose of the pelvis and the places of points on other links. A joint that
a step would take beyond its range is held at the end of the range and the step is solved again.

Left out of the JS: gaps/dist/setBox/setStand/footContacts/jointBeyondLimit/bodyName, which only PickEdit.check
reaches, and check is not ported.

An instance owns one MjData, so it is not safe to share between threads.
"""
import math

import numpy as np

import mujoco

from .mathq import hypot

NV = 35
STAND_PARK = [100, 90]

DEFAULT_WEIGHTS = {
    'foot': 200, 'footRot': 40, 'palm': 200, 'palmRot': 4, 'com': 20,
    'legs': 0.05, 'waist': 0.3, 'arms': 0.05, 'wrists': 0.2, 'lambda': 1e-4,
    'pelvis': 40, 'pelvisRot': 10, 'link': 40, 'jointKey': 5, 'keyHard': 10,
}

_TRIU = np.triu(np.ones((NV, NV)))  # (35, 35)


def _num(v):
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def stand_xml(xml, stand=None):
    """The scene with one more static body, the stand a box can rest on. size: full [x, y, z] in m."""
    s = stand or {'pos': [STAND_PARK[0], STAND_PARK[1], 0.05], 'yaw': 0.0, 'size': [0.1, 0.1, 0.1]}
    q = [math.cos(s['yaw'] / 2), 0, 0, math.sin(s['yaw'] / 2)]
    body = ('<body name="stand" pos="%s" quat="%s">'
            '<geom name="stand" type="box" size="%s" contype="3" conaffinity="3" /></body>'
            % (' '.join(_num(v) for v in s['pos']), ' '.join(_num(v) for v in q),
               ' '.join(_num(v / 2) for v in s['size'])))
    return xml.replace('</worldbody>', body + '</worldbody>')


def posture_weights(w):
    """MuJoCo-order joints: legs 0 to 11, waist 12 to 14, arms 15 to 28 with the wrists as the last three of each."""
    out = np.zeros(29)  # (29,)
    for j in range(29):
        arm = (j - 15) % 7 if j >= 15 else -1
        out[j] = w['legs'] if j < 12 else w['waist'] if j < 15 else (w['wrists'] if arm >= 4 else w['arms'])
    return out


def log_rot(R, Rt):
    """The rotation from R to Rt as a scaled axis, both row-major 3x3 flattened. E = Rt R^T, as in the JS."""
    E = [0.0] * 9
    for i in range(3):
        for j in range(3):
            E[3 * i + j] = Rt[3 * i] * R[3 * j] + Rt[3 * i + 1] * R[3 * j + 1] + Rt[3 * i + 2] * R[3 * j + 2]
    c = max(-1.0, min(1.0, (E[0] + E[4] + E[8] - 1) / 2))
    th = math.acos(c)
    k = 0.5 if th < 1e-6 else th / (2 * math.sin(th))
    return [k * (E[7] - E[5]), k * (E[2] - E[6]), k * (E[3] - E[1])]


class BodyModel:
    def __init__(self, xml, meta):
        self.meta = meta
        self.model = mujoco.MjModel.from_xml_string(stand_xml(xml))
        self.data = mujoco.MjData(self.model)
        m = self.model
        O = mujoco.mjtObj

        def oid(kind, name):
            i = mujoco.mj_name2id(m, kind, name)
            if i < 0:
                raise ValueError('no %s named %s' % (name, kind))
            return i

        joints = [oid(O.mjOBJ_JOINT, n) for n in meta['joints']]
        self.qa = [int(m.jnt_qposadr[j]) for j in joints]
        if any(a != 7 + i for i, a in enumerate(self.qa)):
            raise ValueError('the joints are not qpos 7 to 35 in order')
        self.lo = np.array([float(m.jnt_range[j][0]) for j in joints])  # (29,)
        self.hi = np.array([float(m.jnt_range[j][1]) for j in joints])  # (29,)
        self.feet = [oid(O.mjOBJ_BODY, n) for n in ('left_ankle_roll_link', 'right_ankle_roll_link')]
        self.palms = [oid(O.mjOBJ_SITE, n) for n in meta['palm_sites']]
        self.pelvis = oid(O.mjOBJ_BODY, 'pelvis')
        self.torso = oid(O.mjOBJ_BODY, 'torso_link')
        self.floor = oid(O.mjOBJ_GEOM, 'floor')
        self.box = oid(O.mjOBJ_GEOM, 'box0')
        self.boxQ = int(m.jnt_qposadr[oid(O.mjOBJ_JOINT, 'box0')])
        self.stand = oid(O.mjOBJ_GEOM, 'stand')
        self.standBody = oid(O.mjOBJ_BODY, 'stand')
        self.hands = [oid(O.mjOBJ_GEOM, n) for n in ('left_hand', 'right_hand')]
        box_bodies = {oid(O.mjOBJ_BODY, 'box%d' % i) for i in range(meta['nboxes'])}
        self.body = []
        self.footGeoms = [[], []]
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if g == self.floor or g == self.stand or b in box_bodies or g in self.hands:
                continue
            if not m.geom_contype[g] and not m.geom_conaffinity[g]:
                continue
            self.body.append(g)
            if b in self.feet:
                self.footGeoms[self.feet.index(b)].append(g)
        self.footFlatZ = -min(float(m.geom_pos[g][2]) - float(m.geom_rbound[g]) for g in self.footGeoms[0])
        self.jp = np.zeros((3, m.nv))  # (3, nv)
        self.jr = np.zeros((3, m.nv))  # (3, nv)
        self.A = np.zeros((NV, NV))    # (35, 35)
        self.g = np.zeros(NV)          # (35,)
        self.A0 = np.zeros((NV, NV))   # (35, 35)
        self.g0 = np.zeros(NV)         # (35,)
        self.held = np.zeros(29, dtype=bool)   # (29,)
        self.heldAt = np.zeros(29)             # (29,)
        self.dq = np.zeros(NV)                 # (35,)
        self.row = np.zeros(NV)                # (35,)
        self._outer = np.zeros((NV, NV))       # (35, 35)
        self._di = np.arange(NV)

    def body_id(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def set_robot(self, q36):
        self.data.qpos[0:36] = q36
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def palm(self, s):
        d, i = self.data, self.palms[s]
        return {'p': d.site_xpos[i].tolist(), 'R': d.site_xmat[i].tolist()}

    def foot(self, s):
        d, i = self.data, self.feet[s]
        return {'p': d.xpos[i].tolist(), 'R': d.xmat[i].tolist()}

    def pelvis_pose(self):
        d, i = self.data, self.pelvis
        return {'p': d.xpos[i].tolist(), 'R': d.xmat[i].tolist()}

    def com(self):
        return self.data.subtree_com[self.pelvis].tolist()

    def body_pose(self, i):
        d = self.data
        return {'p': d.xpos[i].tolist(), 'R': d.xmat[i].tolist()}

    def point_world(self, i, at):
        b = self.body_pose(i)
        p, R = b['p'], b['R']
        return [p[k] + R[3 * k] * at[0] + R[3 * k + 1] * at[1] + R[3 * k + 2] * at[2] for k in range(3)]

    def body_heading(self, i):
        R = self.data.xmat[i]
        return math.atan2(-R[1], R[4])

    def targets(self, q36):
        """Every task of a pose, as the targets that keep it."""
        self.set_robot(q36)
        return {'foot': [self.foot(0), self.foot(1)], 'palm': [self.palm(0), self.palm(1)],
                'com': self.com(), 'q': np.array(q36, dtype=np.float64)}

    def add_row(self, e, w):
        wr = w * self.row  # (35,), the JS adds (w * row[a]) * row[b]
        self.g += wr * e
        np.multiply(wr[:, None], self.row[None, :], out=self._outer)
        self._outer *= _TRIU
        self.A += self._outer

    def add_jac(self, buf, e, w):
        for k in range(3):
            self.row[:] = buf[k, :NV]
            self.add_row(e[k], w)

    def cholesky(self):
        A, g, dq = self.A, self.g, self.dq
        for i in range(NV):
            A[i + 1:, i] = A[i, i + 1:]
        for j in range(NV):
            s = A[j, j] - A[j, :j] @ A[j, :j]
            L = math.sqrt(s)
            A[j, j] = L
            if j + 1 < NV:
                A[j + 1:, j] = (A[j + 1:, j] - A[j + 1:, :j] @ A[j, :j]) / L
        for i in range(NV):
            dq[i] = (g[i] - A[i, :i] @ dq[:i]) / A[i, i]
        for i in range(NV - 1, -1, -1):
            dq[i] = (dq[i] - A[i + 1:, i] @ dq[i + 1:]) / A[i, i]

    def step(self, q):
        """The step of the normal equations in A and g, with every joint that it would take beyond its range held
        at the end of the range."""
        A, g, A0, g0, dq, held, heldAt = self.A, self.g, self.A0, self.g0, self.dq, self.held, self.heldAt
        BIG = 1e7
        A0[:] = A
        g0[:] = g
        held.fill(False)
        for rnd in range(5):
            if rnd:
                A[:] = A0
                g[:] = g0
                for j in range(29):
                    if held[j]:
                        a = 6 + j
                        A[a, a] += BIG
                        g[a] += BIG * heldAt[j]
            self.cholesky()
            more = False
            for j in range(29):
                if held[j]:
                    continue
                v = q[7 + j] + dq[6 + j]
                if v > self.hi[j] + 1e-9 or v < self.lo[j] - 1e-9:
                    held[j] = True
                    heldAt[j] = (self.hi[j] if v > self.hi[j] else self.lo[j]) - q[7 + j]
                    more = True
            if not more:
                break

    def integrate(self, q, max_step):
        dq = self.dq
        n = float(np.max(np.abs(dq)))
        k = max_step / n if n > max_step else 1.0
        for a in range(3):
            q[a] += k * dq[a]
        w = [k * dq[3], k * dq[4], k * dq[5]]
        th = hypot(w[0], w[1], w[2])
        s = 0.5 if th < 1e-9 else math.sin(th / 2) / th
        e = [math.cos(th / 2), s * w[0], s * w[1], s * w[2]]
        a, b, c, d = q[3], q[4], q[5], q[6]
        q[3] = a * e[0] - b * e[1] - c * e[2] - d * e[3]
        q[4] = a * e[1] + b * e[0] + c * e[3] - d * e[2]
        q[5] = a * e[2] - b * e[3] + c * e[0] + d * e[1]
        q[6] = a * e[3] + b * e[2] - c * e[1] + d * e[0]
        l = hypot(q[3], q[4], q[5], q[6])
        q[3:7] /= l
        q[7:36] = np.clip(q[7:36] + k * dq[6:35], self.lo, self.hi)
        return k * n

    def solve(self, q, T, W, posture, iters=12, max_step=0.3):
        """q: (36,) float64, changed in place. T: the targets of one frame (see PickEdit.frame_targets)."""
        m, d = self.model, self.data
        A, g = self.A, self.g
        for _ in range(iters):
            self.set_robot(q)
            A.fill(0)
            g.fill(0)
            for s in (0, 1):
                f, h = self.foot(s), self.palm(s)
                mujoco.mj_jacBody(m, d, self.jp, self.jr, self.feet[s])
                tf = T['foot'][s]
                self.add_jac(self.jp, [tf['p'][k] - f['p'][k] for k in range(3)], W['foot'])
                self.add_jac(self.jr, log_rot(f['R'], tf['R']), W['footRot'])
                mujoco.mj_jacSite(m, d, self.jp, self.jr, self.palms[s])
                pw = T['palmW'][s] if T.get('palmW') else 1
                if pw > 0:
                    tp = T['palm'][s]
                    self.add_jac(self.jp, [tp['p'][k] - h['p'][k] for k in range(3)], pw * W['palm'])
                    rw = T['palmRotW'] if T.get('palmRotW') is not None else 1
                    self.add_jac(self.jr, log_rot(h['R'], tp['R']), pw * W['palmRot'] * rw)
            pel = T.get('pelvis')
            if pel and (pel['w'] > 0 or pel['wr'] > 0):
                b = self.pelvis_pose()
                mujoco.mj_jacBody(m, d, self.jp, self.jr, self.pelvis)
                if pel['w'] > 0:
                    self.add_jac(self.jp, [pel['p'][k] - b['p'][k] for k in range(3)], pel['w'] * W['pelvis'])
                if pel['wr'] > 0:
                    self.add_jac(self.jr, log_rot(b['R'], pel['R']), pel['wr'] * W['pelvisRot'])
            for pt in T.get('points') or []:
                if not pt['w'] > 0:
                    continue
                b = self.body_pose(pt['body'])
                at = self.point_world(pt['body'], pt['at'])
                r = [at[k] - b['p'][k] for k in range(3)]
                mujoco.mj_jacBody(m, d, self.jp, self.jr, pt['body'])
                for k in range(3):
                    i, j = (k + 1) % 3, (k + 2) % 3
                    self.row[:] = self.jp[k, :NV] + self.jr[i, :NV] * r[j] - self.jr[j, :NV] * r[i]
                    self.add_row(pt['p'][k] - at[k], pt['w'] * W['link'])
            mujoco.mj_jacSubtreeCom(m, d, self.jp, self.pelvis)
            c = self.com()
            for k in (0, 1):
                self.row[:] = self.jp[k, :NV]
                self.add_row(T['com'][k] - c[k], W['com'])
            A[self._di[6:], self._di[6:]] += posture
            g[6:] += posture * (T['q'][7:36] - q[7:36])
            A[self._di, self._di] += W['lambda']
            self.step(q)
            if self.integrate(q, max_step) < 1e-6:
                break
        return q
