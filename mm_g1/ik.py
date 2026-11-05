"""Port of lmm-g1-box's web/js/ik.js: foot locking after contact_update of orangeduck/Motion-Matching. The G1 leg
is not a two-bone chain, so the ankle position is solved by damped least squares on hip pitch, roll, yaw and
knee, and the ankle pitch and roll then restore the foot orientation of the animation."""
import math

import numpy as np

from .mathq import add, cross, hypot, decay_spring_vec, dot, length, qfrom_angle_axis, scale, sub


def m3_from_quat(a):
    w, x, y, z = a
    return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]


def m3_mul(A, B):
    return [A[0] * B[0] + A[1] * B[3] + A[2] * B[6], A[0] * B[1] + A[1] * B[4] + A[2] * B[7], A[0] * B[2] + A[1] * B[5] + A[2] * B[8],
            A[3] * B[0] + A[4] * B[3] + A[5] * B[6], A[3] * B[1] + A[4] * B[4] + A[5] * B[7], A[3] * B[2] + A[4] * B[5] + A[5] * B[8],
            A[6] * B[0] + A[7] * B[3] + A[8] * B[6], A[6] * B[1] + A[7] * B[4] + A[8] * B[7], A[6] * B[2] + A[7] * B[5] + A[8] * B[8]]


def m3_vec(A, v):
    return [A[0] * v[0] + A[1] * v[1] + A[2] * v[2], A[3] * v[0] + A[4] * v[1] + A[5] * v[2], A[6] * v[0] + A[7] * v[1] + A[8] * v[2]]


def m3_T(A):
    return [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]]


def solve3(A, b):
    a, bb, c, d, e, f, g, h, i = A
    det = a * (e * i - f * h) - bb * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        return [0.0, 0.0, 0.0]
    inv = [(e * i - f * h), -(bb * i - c * h), (bb * f - c * e),
           -(d * i - f * g), (a * i - c * g), -(a * f - c * d),
           (d * h - e * g), -(a * h - bb * g), (a * e - bb * d)]
    return [v / det for v in m3_vec(inv, b)]


LEG = ('hip_pitch_link', 'hip_roll_link', 'hip_yaw_link', 'knee_link', 'ankle_pitch_link', 'ankle_roll_link')


class Leg:
    def __init__(self, bodies, side):
        by_name = {b['name']: b for b in bodies}
        self.chain = [by_name[f'{side}_{n}'] for n in LEG]
        self.local = [{'p': b['pos'], 'R': m3_from_quat(b['quat']), 'axis': b['axis'], 'dof': b['dof']} for b in self.chain]

    def fk(self, pelvis_pos, pelvis_R, joints):
        p, R, out = pelvis_pos, pelvis_R, []
        for L in self.local:
            p = add(p, m3_vec(R, L['p']))
            Rb = m3_mul(R, L['R'])
            R = m3_mul(Rb, m3_from_quat(qfrom_angle_axis(joints[L['dof']], L['axis'])))
            out.append((p, R, m3_vec(Rb, L['axis'])))
        return out

    def solve(self, pelvis_pos, pelvis_R, joints, target, foot_R, iters=6, lam=0.05):
        for _ in range(iters):
            f = self.fk(pelvis_pos, pelvis_R, joints)
            foot = f[5][0]
            e = sub(target, foot)
            if length(e) < 1e-4:
                break
            J = [cross(f[k][2], sub(foot, f[k][0])) for k in range(4)]
            A = [0.0] * 9
            for col in J:
                for i in range(3):
                    for j in range(3):
                        A[3 * i + j] += col[i] * col[j]
            for i in range(3):
                A[4 * i] += lam * lam
            y = solve3(A, e)
            for k in range(4):
                joints[self.local[k]['dof']] += dot(J[k], y)
            knee = self.local[3]['dof']
            joints[knee] = max(joints[knee], 0.02)
        f = self.fk(pelvis_pos, pelvis_R, joints)
        knee_R = f[3][1]
        Rp = m3_mul(knee_R, self.local[4]['R'])
        Rrel = m3_mul(m3_T(m3_mul(Rp, self.local[5]['R'])), foot_R)
        joints[self.local[4]['dof']] = math.atan2(-Rrel[6], Rrel[0])
        joints[self.local[5]['dof']] = math.atan2(-Rrel[5], Rrel[4])


def new_foot():
    return {'state': False, 'lock': False, 'pos': None, 'vel': [0.0, 0.0, 0.0], 'point': [0.0, 0.0, 0.0], 'target': None,
            'offPos': [0.0, 0.0, 0.0], 'offVel': [0.0, 0.0, 0.0], 'final': None}


class FootLock:
    def __init__(self, bodies, unlock_radius=0.2, halflife=0.1):
        self.legs = [Leg(bodies, 'left'), Leg(bodies, 'right')]
        self.unlock_radius = unlock_radius
        self.halflife = halflife
        self.enabled = True
        self.reset()

    def reset(self):
        self.skate = {'dist': 0.0, 'time': 0.0}
        self.feet = [new_foot(), new_foot()]

    def update_contact(self, c, inp, input_state, dt):
        if c['pos'] is None:
            c['pos'], c['target'] = list(inp), list(inp)
        input_vel = scale(sub(inp, c['target']), 1 / dt)
        c['target'] = list(inp)
        c['offPos'], c['offVel'] = decay_spring_vec(c['offPos'], c['offVel'], self.halflife, dt)
        goal, goal_vel = (c['point'], [0.0, 0.0, 0.0]) if c['lock'] else (inp, input_vel)
        c['pos'] = add(goal, c['offPos'])
        c['vel'] = add(goal_vel, c['offVel'])
        unlock = c['lock'] and length(sub(c['point'], inp)) > self.unlock_radius
        if not c['state'] and input_state:
            c['lock'] = True
            c['point'] = list(c['pos'])
            c['offPos'] = sub(add(inp, c['offPos']), c['point'])
            c['offVel'] = add(input_vel, c['offVel'])
        elif (c['lock'] and c['state'] and not input_state) or unlock:
            c['lock'] = False
            c['offPos'] = sub(add(c['point'], c['offPos']), inp)
            c['offVel'] = sub(c['offVel'], input_vel)
        c['state'] = input_state

    def shift(self, d):
        for c in self.feet:
            for k in ('pos', 'point', 'target', 'final'):
                if c[k] is not None:
                    c[k] = [c[k][0] + d[0], c[k][1] + d[1], c[k][2]]

    def turn(self, dyaw, pivot):
        co, s = math.cos(dyaw), math.sin(dyaw)
        pt = lambda p: p and [pivot[0] + co * (p[0] - pivot[0]) - s * (p[1] - pivot[1]),
                              pivot[1] + s * (p[0] - pivot[0]) + co * (p[1] - pivot[1]), p[2]]
        vec = lambda v: v and [co * v[0] - s * v[1], s * v[0] + co * v[1], v[2]]
        for f in self.feet:
            f.update(pos=pt(f['pos']), point=pt(f['point']), target=pt(f['target']), final=pt(f['final']),
                     vel=vec(f['vel']), offPos=vec(f['offPos']), offVel=vec(f['offVel']))

    def apply(self, state, dt):
        """state: render_state() of the controller. Returns the joint angles (29,) with the feet fixed."""
        joints = [float(v) for v in state['joints']]
        pR = m3_from_quat(state['hipRot'])
        for s in range(2):
            leg = self.legs[s]
            foot = leg.fk(state['hipPos'], pR, joints)[5]
            c = self.feet[s]
            self.update_contact(c, foot[0], state['contacts'][s], dt)
            if self.enabled and length(sub(c['pos'], foot[0])) > 1e-4:
                leg.solve(state['hipPos'], pR, joints, c['pos'], foot[1])
            final = leg.fk(state['hipPos'], pR, joints)[5][0]
            if state['contacts'][s] and c['final'] is not None:
                self.skate['dist'] += hypot(final[0] - c['final'][0], final[1] - c['final'][1])
                self.skate['time'] += dt
            c['final'] = final
        return np.asarray(joints)
