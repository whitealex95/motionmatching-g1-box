"""Port of the Controller of lmm-g1-box's web/js/controller.js (exact search only): Locomotion and the box
controller, inertialization, the simulation object and trajectory, root warping and the moving-target rules.

The box controller has four states, set by its tag and by whether the robot holds the box:
  pick, empty      walk in and bend down to the box. The held output of the clip rising is the grasp.
  pick, holding    rise with the box.
  place, holding   walk in and lower the box. The held output falling is the release.
  place, empty     rise without the box.
"""
import copy
import math

import numpy as np

from .database import NDOF, NPOSE_FEATURES, Pose
from .mathq import (add, hypot, damp_factor, decay_spring_quat, decay_spring_scalar, decay_spring_vec, js_sign1, length,
                    norm, qabs, qangle_between, qfrom_angle_axis, qfrom_scaled_angle_axis, qfrom_yaw, qid, qinv,
                    qinv_mul, qinv_mul_vec, qmul, qmul_inv, qmul_vec, qnorm, qslerp_shortest_approx,
                    qto_scaled_angle_axis, qyaw, scale, simple_spring_damper_quat, sub, wrap_angle)

DEFAULTS = dict(
    searchTime=0.1, inertializeHalflife=0.1, simVelocityHalflife=0.27, simRotationHalflife=0.27,
    walkFwd=0.8, walkSide=0.45, walkBack=0.4, slowScale=0.5,
    adjustPosHalflife=0.1, adjustRotHalflife=0.2, adjustPosMaxRatio=0.5, adjustRotMaxRatio=0.5,
    clampMaxDistance=0.15, clampMaxAngle=0.5 * math.pi,
    boxEnter=2.5, boxLeave=4.0, forceEngageRadius=1.1, sideMargin=0.25,
    boxDb='same', loco='nsm', engageRadius=3.0, boxQueryRadius=3.0,
    warpHalflife=0.2, warpMaxRatio=0.8, warpMaxDist=0.7, warpMaxAngle=1.2,
    pickPlaceSpeed=1.0, pickPlaceSlow='clip', warp='always', leaveMoved=None, reapproachTime=1.0,
    boxEntry='walk', entryLead=30, rise='search', transitionCost=0.0,
    zLow=0.66, zUpright=0.72, uprightTilt=None,
    graspOn=0.6, releaseOff=0.4, graspReach=0.25, floorBoxLift=0.15,
    holdSwitchMax=0.5, movedTolerance=0.12, movedAngle=0.35, collisionRadius=0.36,
    standOff=0.36, approachClearance=0.3, approachFallback=0.6,
)
QUERY_DB = {'loco': 'loco', 'pick': 'box', 'place': 'box'}


def torso_chain(bodies):
    chain = []
    k = next(i for i, b in enumerate(bodies) if b['name'] == 'torso_link')
    while k >= 0:
        chain.insert(0, bodies[k])
        k = bodies[k]['parent']
    return chain


def torso_tilt(chain, hip_rot, joints):
    """Torso link tilt from vertical (deg)."""
    r = qnorm(hip_rot)
    for body in chain:
        if body['parent'] < 0:
            continue
        r = qmul(r, body['quat'])
        if body.get('axis'):
            r = qmul(r, qfrom_angle_axis(joints[body['dof']], body['axis']))
    return math.acos(max(-1.0, min(1.0, qmul_vec(r, [0.0, 0.0, 1.0])[2]))) * 180 / math.pi


def target_sides(pos, yaw, n):
    """The n sides a box is picked up from: side k turns its X axis by k / n of a full turn."""
    out = []
    for side in range(n):
        y = yaw + side * 2 * math.pi / n
        out.append({'side': side, 'yaw': y, 'dir': [math.cos(y), math.sin(y)], 'pos': list(pos[:2])})
    return out


def zero_offsets():
    return {'rootPos': [0.0, 0.0, 0.0], 'rootVel': [0.0, 0.0, 0.0], 'rootRot': qid(), 'rootAng': [0.0, 0.0, 0.0],
            'hipPos': [0.0, 0.0, 0.0], 'hipVel': [0.0, 0.0, 0.0], 'hipRot': qid(), 'hipAng': [0.0, 0.0, 0.0],
            'joints': np.zeros(NDOF), 'jointVel': np.zeros(NDOF),
            'boxPos': [0.0, 0.0, 0.0], 'boxVel': [0.0, 0.0, 0.0], 'boxRot': qid(), 'boxAng': [0.0, 0.0, 0.0]}


class Controller:
    STATE = ('mode', 'boxTag', 'holding', 'sim', 'desiredRot', 'desiredVel', 'searchTimer', 'forceSearch',
             'holdSwitchTime', 'commit', 'target', 'boxCost', 'lastCosts', 'calledOff', 'engaged', 'reapproach',
             'leaveBox', 'missed', 'walkIn', 'semantics', 'events', 'trajectory', 'curr', 'pose', 'off', 'tsrc',
             'tdst', 'navigating')

    def __init__(self, cfg, backend, bodies, **opts):
        self.cfg = cfg
        self.p = dict(DEFAULTS)
        self.p.update(opts)
        self.p['weights'], self.p['gains'] = {}, {}
        if 'uprightTilt' not in opts:
            self.p['uprightTilt'] = cfg.get('upright_tilt_deg')
        for key, ck in (('warp', 'warp'), ('leaveMoved', 'leave_moved_m'), ('rise', 'rise'), ('transitionCost', 'transition_cost')):
            if key not in opts and ck in cfg:
                self.p[key] = cfg[ck]
        self.torso = torso_chain(bodies)
        self.p['floorBoxZ'] = cfg['box_size'][2] / 2 + self.p['floorBoxLift']
        self.p['placeEngageRadius'] = cfg.get('place_engage_radius', self.p['forceEngageRadius'])
        self.p['placeEntryFrames'] = cfg.get('place_entry_frames', 0)
        self.dt = 1.0 / cfg['fps']
        for kind, db in QUERY_DB.items():
            self.set_weights(kind, cfg['controllers'][db]['weights'])
        self.backend = backend
        self.navigating = False
        self.lastMatch = {}
        self.reset([0.0, 0.0, 0.0], 0.0, False)

    def set_weights(self, kind, weights):
        c = self.cfg['controllers'][QUERY_DB[kind]]
        gain = np.zeros(c['nfeatures'])
        i = 0
        for g, n in enumerate(c['group_sizes']):
            gain[i:i + n] = (weights[g] / c['weights'][g]) ** 2
            i += n
        self.p['weights'][kind] = list(weights)
        self.p['gains'][kind] = gain

    def reset(self, pos, yaw, holding=False):
        rot = qfrom_yaw(yaw)
        self.mode = 'loco'
        self.boxTag = None
        self.holding = holding
        self.sim = {'pos': list(pos), 'vel': [0.0, 0.0, 0.0], 'acc': [0.0, 0.0, 0.0], 'rot': list(rot), 'ang': [0.0, 0.0, 0.0]}
        self.desiredRot = list(rot)
        self.desiredVel = [0.0, 0.0, 0.0]
        self.searchTimer = 0.0
        self.forceSearch = True
        self.holdSwitchTime = 0.0
        self.commit = None
        self.target = None
        self.boxCost = math.inf
        self.lastCosts = []
        self.lastMatch = {}
        self.calledOff = False
        self.engaged = None
        self.reapproach = None
        self.leaveBox = False
        self.missed = False
        self.walkIn = False
        self.semantics = 'none'
        self.events = {'grasp': False, 'release': False}
        self.trajectory = {'pos': [list(pos) for _ in range(4)], 'rot': [list(rot) for _ in range(4)]}
        b = self.backend
        m = self.match('loco', self.standing_query(holding))
        b.commit(m)
        p = b.match_pose(m, list(pos), list(rot), self.dt)
        p.rootPos, p.rootRot = list(pos), list(rot)
        self.curr = p
        self.pose = p.copy()
        self.off = zero_offsets()
        self.tsrc = {'pos': list(pos), 'rot': list(rot)}
        self.tdst = {'pos': list(pos), 'rot': list(rot)}

    def standing_query(self, holding):
        c = self.backend.ctrl['loco']
        qy = np.asarray(c['offset'], dtype=np.float32).copy()
        qy[6:NPOSE_FEATURES + 12] = 0
        for h in range(3):
            qy[NPOSE_FEATURES + 6 + 2 * h] = 1
        qy[NPOSE_FEATURES + 12] = 0 if holding else 1
        qy[NPOSE_FEATURES + 13] = 1 if holding else 0
        return qy

    # ---- snapshot for the replan (Sim.saveState of the page: the databases and settings are shared) ----

    def save(self):
        return copy.deepcopy({k: getattr(self, k) for k in self.STATE}), self.backend.state()

    def restore(self, s):
        state, bs = s
        for k, v in state.items():
            setattr(self, k, v)
        self.backend.set_state(bs)

    # ---- inertialization (inertialize_pose_transition / _update / _root_adjust) ----

    def transition(self, dst):
        P, C, O = self.pose, self.curr, self.off
        self.tdst = {'pos': list(P.rootPos), 'rot': list(P.rootRot)}
        self.tsrc = {'pos': list(dst.rootPos), 'rot': list(dst.rootRot)}
        to_world = lambda v: qmul_vec(self.tdst['rot'], qinv_mul_vec(self.tsrc['rot'], v))
        wdv, wda = to_world(dst.rootVel), to_world(dst.rootAng)
        O['rootVel'] = sub(add(P.rootVel, O['rootVel']), wdv)
        O['rootAng'] = sub(add(O['rootAng'], P.rootAng), wda)
        O['hipPos'] = sub(add(C.hipPos, O['hipPos']), dst.hipPos)
        O['hipVel'] = sub(add(C.hipVel, O['hipVel']), dst.hipVel)
        O['hipRot'] = qabs(qmul(qmul(O['hipRot'], C.hipRot), qinv(dst.hipRot)))
        O['hipAng'] = sub(add(O['hipAng'], C.hipAng), dst.hipAng)
        O['boxPos'] = sub(add(C.boxPos, O['boxPos']), dst.boxPos)
        O['boxRot'] = qabs(qmul(qmul(O['boxRot'], C.boxRot), qinv(dst.boxRot)))
        O['joints'] = C.joints + O['joints'] - dst.joints
        O['jointVel'] = C.jointVel + O['jointVel'] - dst.jointVel
        self.curr = dst

    def inertialize_update(self):
        P, C, O, h, dt = self.pose, self.curr, self.off, self.p['inertializeHalflife'], self.dt
        tdr, tsr = self.tdst['rot'], self.tsrc['rot']
        wpos = add(qmul_vec(tdr, qinv_mul_vec(tsr, sub(C.rootPos, self.tsrc['pos']))), self.tdst['pos'])
        wvel = qmul_vec(tdr, qinv_mul_vec(tsr, C.rootVel))
        wrot = qnorm(qmul(tdr, qinv_mul(tsr, C.rootRot)))
        wang = qmul_vec(tdr, qinv_mul_vec(tsr, C.rootAng))
        O['rootPos'], O['rootVel'] = decay_spring_vec(O['rootPos'], O['rootVel'], h, dt)
        P.rootPos, P.rootVel = add(wpos, O['rootPos']), add(wvel, O['rootVel'])
        O['rootRot'], O['rootAng'] = decay_spring_quat(O['rootRot'], O['rootAng'], h, dt)
        P.rootRot = qnorm(qmul(O['rootRot'], wrot))
        P.rootAng = add(O['rootAng'], qmul_vec(O['rootRot'], wang))
        O['hipPos'], O['hipVel'] = decay_spring_vec(O['hipPos'], O['hipVel'], h, dt)
        P.hipPos, P.hipVel = add(C.hipPos, O['hipPos']), add(C.hipVel, O['hipVel'])
        O['hipRot'], O['hipAng'] = decay_spring_quat(O['hipRot'], O['hipAng'], h, dt)
        P.hipRot = qnorm(qmul(O['hipRot'], C.hipRot))
        P.hipAng = add(O['hipAng'], qmul_vec(O['hipRot'], C.hipAng))
        O['boxPos'], O['boxVel'] = decay_spring_vec(O['boxPos'], O['boxVel'], h, dt)
        P.boxPos = add(C.boxPos, O['boxPos'])
        O['boxRot'], O['boxAng'] = decay_spring_quat(O['boxRot'], O['boxAng'], h, dt)
        P.boxRot = qnorm(qmul(O['boxRot'], C.boxRot))
        O['joints'], O['jointVel'] = decay_spring_scalar(O['joints'], O['jointVel'], h, dt)
        P.joints = C.joints + O['joints']
        P.jointVel = C.jointVel + O['jointVel']
        P.contacts = list(C.contacts)
        P.held = C.held

    def root_adjust(self, input_pos, input_rot):
        P, O = self.pose, self.off
        dpos = sub(input_pos, P.rootPos)
        P.rootPos = add(P.rootPos, dpos)
        self.tdst['pos'] = add(self.tdst['pos'], dpos)
        self.tsrc['pos'] = add(self.tsrc['pos'], qmul_vec(self.tsrc['rot'], qinv_mul_vec(
            self.tdst['rot'], sub(sub(P.rootPos, O['rootPos']), self.tdst['pos']))))
        self.tdst['pos'] = list(P.rootPos)
        O['rootPos'] = [0.0, 0.0, 0.0]
        drot = qnorm(qmul_inv(input_rot, P.rootRot))
        P.rootRot = qmul(drot, P.rootRot)
        self.tdst['rot'] = qmul(drot, self.tdst['rot'])

    def shift(self, d):
        """Moves the character, its simulation object and its trajectory by d (x, y): the anchoring."""
        P = self.pose
        self.root_adjust([P.rootPos[0] + d[0], P.rootPos[1] + d[1], P.rootPos[2]], P.rootRot)
        self.sim['pos'] = [self.sim['pos'][0] + d[0], self.sim['pos'][1] + d[1], self.sim['pos'][2]]
        for p in self.trajectory['pos']:
            p[0] += d[0]
            p[1] += d[1]

    def turn(self, dyaw, pivot):
        """Turns the character, its simulation object and its trajectory by dyaw about the vertical line through
        pivot (x, y): the snap-xyyaw anchoring. Box targets stay in the world."""
        c, s, r = math.cos(dyaw), math.sin(dyaw), qfrom_yaw(dyaw)
        pt = lambda p: [pivot[0] + c * (p[0] - pivot[0]) - s * (p[1] - pivot[1]),
                        pivot[1] + s * (p[0] - pivot[0]) + c * (p[1] - pivot[1]), p[2]]
        vec = lambda v: [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]
        P = self.pose
        self.root_adjust(pt(P.rootPos), qmul(r, P.rootRot))
        P.rootVel = vec(P.rootVel)
        self.off['rootVel'] = vec(self.off['rootVel'])
        self.sim.update(pos=pt(self.sim['pos']), vel=vec(self.sim['vel']), acc=vec(self.sim['acc']), rot=qmul(r, self.sim['rot']))
        self.desiredRot = qmul(r, self.desiredRot)
        self.desiredVel = vec(self.desiredVel)
        self.trajectory['pos'] = [pt(p) for p in self.trajectory['pos']]
        self.trajectory['rot'] = [qmul(r, x) for x in self.trajectory['rot']]

    # ---- simulation object and trajectory ----

    def desired_velocity(self, stick, sim_rot, speed):
        if not stick:
            return [0.0, 0.0, 0.0]
        local = qinv_mul_vec(sim_rot, [stick[0], stick[1], 0.0])
        s = [local[0] * (speed['fwd'] if local[0] > 0 else speed['back']), local[1] * speed['side'], 0.0]
        return qmul_vec(sim_rot, s)

    def sim_position_update(self, s, desired_vel, halflife, dt, obstacles):
        y = (4 * math.log(2.0)) / (halflife + 1e-5) / 2
        j0 = sub(s['vel'], desired_vel)
        j1 = add(s['acc'], scale(j0, y))
        e = math.exp(-y * dt)
        prev = s['pos']
        pos = add(add(add(
            scale(add(scale(j1, -1 / (y * y)), scale(sub(scale(j0, -1), scale(j1, dt)), 1 / y)), e),
            scale(j1, 1 / (y * y))), scale(j0, 1 / y)), add(scale(desired_vel, dt), prev))
        s['vel'] = add(scale(add(j0, scale(j1, dt)), e), desired_vel)
        s['acc'] = scale(sub(s['acc'], scale(j1, y * dt)), e)
        s['pos'] = self.collide(pos, obstacles)

    def collide(self, pos, obstacles):
        p = list(pos)
        r = self.p['collisionRadius']
        for o in obstacles:
            c, s = math.cos(o['yaw']), math.sin(o['yaw'])
            dx, dy = p[0] - o['pos'][0], p[1] - o['pos'][1]
            lx, ly = c * dx + s * dy, -s * dx + c * dy
            hx, hy = o['half'][0] + r, o['half'][1] + r
            if abs(lx) < hx and abs(ly) < hy:
                nx, ny = lx, ly
                if hx - abs(lx) < hy - abs(ly):
                    nx = js_sign1(lx) * hx
                else:
                    ny = js_sign1(ly) * hy
                p = [o['pos'][0] + c * nx - s * ny, o['pos'][1] + s * nx + c * ny, 0.0]
        return p

    def steer(self, frm, direction, goal_dist, obstacles):
        d = list(direction)
        for o in obstacles:
            rel = [o['pos'][0] - frm[0], o['pos'][1] - frm[1]]
            dist = hypot(rel[0], rel[1])
            along = rel[0] * direction[0] + rel[1] * direction[1]
            if dist > 1.6 or along < 0 or along > goal_dist:
                continue
            lateral = direction[0] * rel[1] - direction[1] * rel[0]
            clearance = max(o['half'][0], o['half'][1]) + self.p['collisionRadius'] + 0.15
            if abs(lateral) >= clearance:
                continue
            side = -1 if lateral > 0 else 1
            w = 2 * (1 - abs(lateral) / clearance) * (1 - dist / 1.6)
            d = [d[0] - side * w * direction[1], d[1] + side * w * direction[0]]
        l = hypot(d[0], d[1]) or 1.0
        return [d[0] / l, d[1] / l]

    def predict_trajectory(self, stick, speed, obstacles):
        sim, dt = self.sim, 20 * self.dt
        rots, drot, dvel = [list(sim['rot'])], [list(self.desiredRot)], [list(self.desiredVel)]
        for i in range(1, 4):
            dv = self.desired_velocity(stick, rots[i - 1], speed)
            drot.append(qfrom_yaw(math.atan2(dv[1], dv[0])) if stick else drot[i - 1])
            x, _ = simple_spring_damper_quat(list(sim['rot']), list(sim['ang']), drot[i], self.p['simRotationHalflife'], i * dt)
            rots.append(x)
            dvel.append(self.desired_velocity(stick, rots[i], speed))
        pos = [list(sim['pos'])]
        s = {'pos': list(sim['pos']), 'vel': list(sim['vel']), 'acc': list(sim['acc'])}
        for i in range(1, 4):
            self.sim_position_update(s, dvel[i], self.p['simVelocityHalflife'], dt, obstacles)
            pos.append(list(s['pos']))
        self.trajectory = {'pos': pos, 'rot': rots}

    # ---- queries ----

    def pose_features(self):
        b = self.backend
        return b.denormalize(b.active, b.current_features())[:NPOSE_FEATURES]

    def loco_query(self, holding):
        qy = np.zeros(self.backend.ctrl['loco']['nfeatures'], dtype=np.float32)
        qy[:NPOSE_FEATURES] = self.pose_features()
        P = self.pose
        for h in range(3):
            tp = qinv_mul_vec(P.rootRot, sub(self.trajectory['pos'][h + 1], P.rootPos))
            td = qinv_mul_vec(P.rootRot, qmul_vec(self.trajectory['rot'][h + 1], [1.0, 0.0, 0.0]))
            qy[NPOSE_FEATURES + 2 * h], qy[NPOSE_FEATURES + 2 * h + 1] = tp[0], tp[1]
            qy[NPOSE_FEATURES + 6 + 2 * h], qy[NPOSE_FEATURES + 6 + 2 * h + 1] = td[0], td[1]
        qy[NPOSE_FEATURES + 12] = 0 if holding else 1
        qy[NPOSE_FEATURES + 13] = 1 if holding else 0
        return qy

    def box_query(self, target_pos, target_dir, tag):
        qy = np.zeros(self.backend.ctrl['box']['nfeatures'], dtype=np.float32)
        qy[:NPOSE_FEATURES] = self.pose_features()
        P = self.pose
        tp = qinv_mul_vec(P.rootRot, [target_pos[0] - P.rootPos[0], target_pos[1] - P.rootPos[1], 0.0])
        r = hypot(tp[0], tp[1])
        if r > self.p['boxQueryRadius']:
            tp = scale(tp, self.p['boxQueryRadius'] / r)
        td = qinv_mul_vec(P.rootRot, [target_dir[0], target_dir[1], 0.0])
        qy[NPOSE_FEATURES], qy[NPOSE_FEATURES + 1] = tp[0], tp[1]
        qy[NPOSE_FEATURES + 2], qy[NPOSE_FEATURES + 3] = td[0], td[1]
        qy[NPOSE_FEATURES + 4] = 1 if tag == 'pick' else 0
        qy[NPOSE_FEATURES + 5] = 1 if tag == 'place' else 0
        qy[NPOSE_FEATURES + 6] = 0 if self.holding else 1
        qy[NPOSE_FEATURES + 7] = 1 if self.holding else 0
        hold = self.anim_hold()
        if self.bent_for_event() and hold is not None:
            qy[NPOSE_FEATURES + 6], qy[NPOSE_FEATURES + 7] = hold[0], hold[1]
        return qy

    def anim_hold(self):
        b = self.backend
        if b.active == 'loco':
            return None
        f = b.denormalize(b.active, b.current_features())
        return [f[NPOSE_FEATURES + 6], f[NPOSE_FEATURES + 7]]

    def bent_for_event(self):
        ph = self.box_phase()
        return ph in ('pick-pre', 'place-pre') and self.is_low()

    def anim_target(self):
        b = self.backend
        if b.active == 'loco':
            return None
        f = b.denormalize(b.active, b.current_features())
        P = self.pose
        pos = add(P.rootPos, qmul_vec(P.rootRot, [float(f[NPOSE_FEATURES]), float(f[NPOSE_FEATURES + 1]), 0.0]))
        d = qmul_vec(P.rootRot, [float(f[NPOSE_FEATURES + 2]), float(f[NPOSE_FEATURES + 3]), 0.0])
        return {'pos': pos, 'dir': d}

    def held_box_world(self):
        P = self.pose
        return {'pos': add(P.rootPos, qmul_vec(P.rootRot, P.boxPos)), 'rot': qnorm(qmul(P.rootRot, P.boxRot))}

    def hip_height(self):
        return self.pose.hipPos[2]

    def is_low(self):
        return self.hip_height() < self.p['zLow']

    def is_upright(self):
        return self.hip_height() > self.p['zUpright']

    def after_last_step(self):
        b = self.backend
        return self.mode == 'box' and b.planted(b.active, b.frame)

    def clip_speed(self, name, frame=None):
        b = self.backend
        if name == 'loco':
            return 1.0
        frame = b.frame if frame is None else frame
        return 1.0 if self.p['pickPlaceSlow'] == 'step' and not b.planted(name, frame) else self.p['pickPlaceSpeed']

    def torso_tilt(self):
        return torso_tilt(self.torso, self.pose.hipRot, self.pose.joints)

    def risen(self):
        return self.is_upright() and (self.p['uprightTilt'] is None or self.torso_tilt() < self.p['uprightTilt'])

    def rising(self):
        b = self.backend
        return (self.p['rise'] == 'finish' and self.mode == 'box' and b.active != 'loco'
                and b.after_event(b.active)[b.frame] == 1)

    def box_phase(self):
        if self.mode != 'box':
            return None
        if self.boxTag == 'pick':
            return 'pick-post' if self.holding else 'pick-pre'
        return 'place-pre' if self.holding else 'place-post'

    def call_off(self):
        self.calledOff = True
        self.forceSearch = True

    # ---- the frame update ----

    def update(self, inp):
        """inp: stick [x, y] world direction or None, slow, holding, pickBox {pos, yaw, id} or None,
        placeTarget {pos, yaw} or None, boxes (floor boxes [{pos, yaw, id}]), boxHalf [x, y], forceSearch."""
        dt, p, b = self.dt, self.p, self.backend
        self.holding = inp['holding']
        self.events = {'grasp': False, 'release': False}
        ss = p['slowScale'] if inp.get('slow') else 1.0
        speed = {'fwd': p['walkFwd'] * ss, 'side': p['walkSide'] * ss, 'back': p['walkBack'] * ss}
        goal = inp['placeTarget'] if inp['holding'] else inp['pickBox']
        activity = ('place' if inp['placeTarget'] else None) if inp['holding'] else ('pick' if inp['pickBox'] else None)
        pick_box = inp['pickBox']
        obstacles = [{'id': o['id'], 'pos': o['pos'], 'yaw': o['yaw'], 'half': inp['boxHalf']} for o in inp['boxes']
                     if not (self.mode == 'box' and pick_box and o['id'] == pick_box['id'])]

        stick = inp.get('stick')
        navigating = False
        if not stick and goal and self.mode == 'loco':
            d = [goal['pos'][0] - self.sim['pos'][0], goal['pos'][1] - self.sim['pos'][1]]
            dist = hypot(d[0], d[1])
            if dist > 0.05:
                mag = min(1.0, max(0.35, (dist - 0.4) / 1.0))
                dr = self.steer(self.sim['pos'], [d[0] / dist, d[1] / dist], dist, obstacles)
                stick = [mag * dr[0], mag * dr[1]]
                navigating = True
        self.navigating = navigating

        dvel = self.desired_velocity(stick, self.sim['rot'], speed)
        dvel_change = length(sub(dvel, self.desiredVel)) / dt
        self.desiredVel = dvel
        if stick and hypot(stick[0], stick[1]) > 0.01:
            self.desiredRot = qfrom_yaw(math.atan2(dvel[1], dvel[0]))
        self.predict_trajectory(stick, speed, obstacles)
        if inp.get('forceSearch'):
            self.forceSearch = True
        if not navigating and dvel_change > 50 * 0.25:
            self.forceSearch = True

        sides = target_sides(goal['pos'], goal['yaw'], self.cfg['box_sides']) if goal else []
        phase = self.box_phase()
        if self.commit and self.mode == 'loco':
            self.commit = None

        pre = phase in ('pick-pre', 'place-pre')
        if not pre or not goal:
            self.engaged = None
        elif not self.engaged or self.engaged['phase'] != phase:
            self.engaged = {'phase': phase, 'pos': list(goal['pos'])}
        if self.reapproach is not None:
            self.reapproach += dt
            if self.reapproach > p['reapproachTime'] or not activity:
                self.reapproach = None
        self.leaveBox = bool(pre and self.is_upright() and p['leaveMoved'] is not None and self.engaged
                             and hypot(goal['pos'][0] - self.engaged['pos'][0],
                                            goal['pos'][1] - self.engaged['pos'][1]) > p['leaveMoved'])
        rising = self.rising()
        self.missed = bool(rising and pre and (self.risen() or b.end_of_anim()))
        if self.leaveBox or self.missed:
            self.forceSearch = True
        if self.walkIn and (not activity or (self.mode == 'box' and (not pre or self.after_last_step()))):
            self.walkIn = False

        abort = False
        if pre and not rising and not self.is_upright() and (self.commit or self.target):
            t = self.target
            c = self.commit or {'boxId': pick_box['id'] if phase == 'pick-pre' and pick_box else None,
                                'side': t['side'], 'pos': list(t['pos']), 'yaw': t['yaw'], 'dir': list(t['dir'])}
            want = 'pick' if phase == 'pick-pre' else 'place'
            abort = bool(self.calledOff or activity != want or inp.get('stick')
                         or (pick_box and phase == 'pick-pre' and c['boxId'] != pick_box['id'])
                         or hypot(sides[c['side']]['pos'][0] - c['pos'][0], sides[c['side']]['pos'][1] - c['pos'][1]) > p['movedTolerance']
                         or abs(wrap_angle(sides[c['side']]['yaw'] - c['yaw'])) > p['movedAngle']) if sides else True
            if abort and not self.commit:
                self.commit = c
        self.calledOff = False
        if abort:
            self.boxTag = 'place' if self.boxTag == 'pick' else 'pick'
            self.forceSearch = True

        hold = self.anim_hold()
        self.holdSwitchTime = self.holdSwitchTime + dt if (self.bent_for_event() and hold is not None and 0.1 < hold[1] < 0.9) else 0.0
        finishing = rising and not self.risen() and not b.end_of_anim()
        search_due = not finishing and (self.forceSearch or self.searchTimer <= 0 or b.end_of_anim())
        if search_due:
            self.search(inp, sides, activity, goal, self.forceSearch, obstacles)
            self.searchTimer = p['searchTime'] / self.clip_speed(b.active)
            self.forceSearch = False
        self.searchTimer -= dt

        ph = self.box_phase()
        if ph in ('pick-pre', 'place-pre') and not self.commit and self.target and self.is_low():
            self.commit = {'boxId': pick_box['id'] if pick_box else None, 'side': self.target['side'],
                           'pos': list(self.target['pos']), 'yaw': self.target['yaw'], 'dir': list(self.target['dir'])}

        b.step(dt)
        self.curr = b.current_pose(self.curr.rootPos, self.curr.rootRot, dt, self.clip_speed(b.active))
        self.inertialize_update()

        held = self.curr.held
        hb = self.held_box_world()
        if (ph == 'pick-pre' and held >= p['graspOn'] and pick_box
                and hypot(hb['pos'][0] - pick_box['pos'][0], hb['pos'][1] - pick_box['pos'][1]) < p['graspReach']):
            self.events['grasp'] = True
            if not self.commit and self.target:
                self.commit = {'boxId': pick_box['id'], 'side': self.target['side'], 'pos': list(self.target['pos']),
                               'yaw': self.target['yaw'], 'dir': list(self.target['dir'])}
        if ph == 'place-pre' and held <= p['releaseOff'] and hb['pos'][2] < p['floorBoxZ']:
            self.events['release'] = True
            if not self.commit and self.target:
                self.commit = {'boxId': None, 'side': self.target['side'], 'pos': list(self.target['pos']),
                               'yaw': self.target['yaw'], 'dir': list(self.target['dir'])}

        if self.mode == 'loco':
            self.sim_position_update(self.sim, self.desiredVel, p['simVelocityHalflife'], dt, obstacles)
            self.sim['rot'], self.sim['ang'] = simple_spring_damper_quat(self.sim['rot'], self.sim['ang'], self.desiredRot,
                                                                         p['simRotationHalflife'], dt)
            self.adjust_to_simulation()
        else:
            if ph in ('pick-pre', 'place-pre') and not ((p['warp'] == 'step' or (p['warp'] == 'pickstep' and ph == 'pick-pre'))
                                                        and self.after_last_step()):
                self.warp_to_target()
            self.sim['pos'] = list(self.pose.rootPos)
            self.sim['vel'] = list(self.pose.rootVel)
            self.sim['acc'] = [0.0, 0.0, 0.0]
            self.sim['rot'] = list(self.pose.rootRot)
            self.sim['ang'] = [0.0, 0.0, 0.0]
            self.desiredRot = list(self.pose.rootRot)
        self.semantics = ('navigate' if navigating else 'locomotion') if self.mode == 'loco' else self.box_phase()

    def approach_clear(self, frm, side, obstacles):
        p = self.p
        stand = [side['pos'][0] - p['standOff'] * side['dir'][0], side['pos'][1] - p['standOff'] * side['dir'][1]]
        d = [stand[0] - frm[0], stand[1] - frm[1]]
        len2 = d[0] * d[0] + d[1] * d[1] or 1e-9
        for o in obstacles:
            t = max(0.0, min(1.0, ((o['pos'][0] - frm[0]) * d[0] + (o['pos'][1] - frm[1]) * d[1]) / len2))
            gap = hypot(frm[0] + t * d[0] - o['pos'][0], frm[1] + t * d[1] - o['pos'][1])
            if not gap > hypot(o['half'][0], o['half'][1]) + p['approachClearance']:
                return False
        return True

    def set_loco(self, name):
        b = self.backend
        if name not in b.locos or b.loco == name:
            return
        qy = self.loco_query(self.holding) if b.active == 'loco' else None
        b.use_loco(name)
        self.p['loco'] = name
        if qy is None:
            return
        m = self.match('loco', qy)
        self.transition(b.match_pose(m, self.curr.rootPos, self.curr.rootRot, self.dt))
        b.commit(m)

    def db_of(self, kind):
        return kind if kind != 'loco' and self.p['boxDb'] == 'different' else QUERY_DB[kind]

    def match(self, kind, query, entry=0):
        b, name = self.backend, self.db_of(kind)
        qn, gain = b.normalize(name, query), self.p['gains'][kind]
        sb_pre = (kind != 'loco' and self.cfg.get('scenebot_entry_frames') and b.active == name
                  and b.scenebot_pre(name)[b.frame] == 1)
        mask = None
        if kind != 'loco' and self.p['boxEntry'] == 'walk':
            mask = b.before_event_mask(name) if sb_pre else b.walk_in_mask(name) if self.walkIn else b.no_entry(name)
        m = b.match(name, qn, gain, entry=entry, mask=mask, transition_cost=self.p['transitionCost'])
        m.update(kind=kind)
        return m

    def search(self, inp, sides, activity, goal, forced, obstacles):
        b, p, P = self.backend, self.p, self.pose
        enter, leave, margin = p['boxEnter'], p['boxLeave'], p['sideMargin']
        self.lastCosts = []
        box_match = None
        phase = self.box_phase()
        pick_box = inp['pickBox']

        if phase in ('pick-post', 'place-post'):
            c = self.commit or self.target
            if c and not self.risen():
                box_match = self.match(self.boxTag, self.box_query(c['pos'], c['dir'], self.boxTag))
        elif activity and sides:
            near = hypot(goal['pos'][0] - P.rootPos[0], goal['pos'][1] - P.rootPos[1])
            if self.mode == 'box' or near < p['engageRadius']:
                locked = self.mode == 'box' and self.target and (self.is_low() or self.commit)
                others = [o for o in obstacles if not pick_box or o['id'] != pick_box['id']]
                cands = [sides[self.target['side']]] if locked else [s for s in sides if self.approach_clear(P.rootPos, s, others)]
                if not cands and near < p['approachFallback']:
                    cands = sides
                best = None
                entry = p['placeEntryFrames'] if self.mode == 'loco' and activity == 'place' else 0
                bonus = lambda x: margin if self.target and x['side'] == self.target['side'] else 0.0
                for s in cands:
                    m = self.match(activity, self.box_query(s['pos'], s['dir'], activity), entry)
                    self.lastCosts.append(m['cost'])
                    if best is None or m['cost'] - bonus(s) < best[1]['cost'] - bonus(best[0]):
                        best = (s, m)
                if best:
                    box_match = best[1]
                    self.target = best[0]
        self.boxCost = box_match['cost'] if box_match else math.inf
        if box_match:
            self.lastMatch[box_match['kind']] = box_match

        use_box = False
        engage = p['placeEngageRadius'] if activity == 'place' else p['forceEngageRadius']
        if self.leaveBox or self.missed:
            self.reapproach = 0.0
            self.engaged = None
            self.walkIn = True
        elif self.mode == 'box':
            if phase in ('pick-post', 'place-post'):
                use_box = bool(box_match)
            elif not activity or inp.get('stick'):
                use_box = not self.is_upright() and bool(box_match)
            else:
                dist = hypot(goal['pos'][0] - P.rootPos[0], goal['pos'][1] - P.rootPos[1])
                use_box = bool(box_match) and not (self.is_upright() and self.boxCost > leave and dist > engage + 0.3)
        elif box_match and not inp.get('stick'):
            near = hypot(goal['pos'][0] - P.rootPos[0], goal['pos'][1] - P.rootPos[1]) < engage
            use_box = self.boxCost < enter or (near and self.reapproach is None)
            if use_box:
                self.boxTag = activity
                self.reapproach = None

        if use_box:
            m = box_match
            self.mode = 'box'
        else:
            if self.mode == 'box':
                self.commit = None
                self.boxTag = None
            self.mode = 'loco'
            if not activity:
                self.target = None
            m = self.match('loco', self.loco_query(inp['holding']))
            self.lastMatch['loco'] = m
        if m['same']:
            return
        dst = b.match_pose(m, self.curr.rootPos, self.curr.rootRot, self.dt, self.clip_speed(m['name'], m['frame']))
        self.transition(dst)
        b.commit(m)

    def adjust_to_simulation(self):
        p, P, dt, sim = self.p, self.pose, self.dt, self.sim
        adj_pos = scale(sub(sim['pos'], P.rootPos), damp_factor(p['adjustPosHalflife'], dt))
        max_len = p['adjustPosMaxRatio'] * length(P.rootVel) * dt
        if length(adj_pos) > max_len:
            adj_pos = scale(norm(adj_pos), max_len)
        pos = add(adj_pos, P.rootPos)
        diff = qabs(qnorm(qmul_inv(sim['rot'], P.rootRot)))
        adj_rot = qslerp_shortest_approx(qid(), diff, damp_factor(p['adjustRotHalflife'], dt))
        max_ang = p['adjustRotMaxRatio'] * length(P.rootAng) * dt
        aa = qto_scaled_angle_axis(adj_rot)
        if length(aa) > max_ang:
            adj_rot = qfrom_scaled_angle_axis(scale(norm(aa), max_ang))
        rot = qmul(adj_rot, P.rootRot)
        if length(sub(pos, sim['pos'])) > p['clampMaxDistance']:
            pos = add(scale(norm(sub(pos, sim['pos'])), p['clampMaxDistance']), sim['pos'])
        if qangle_between(rot, sim['rot']) > p['clampMaxAngle']:
            d = qabs(qmul_inv(rot, sim['rot']))
            ang = max(-p['clampMaxAngle'], min(p['clampMaxAngle'], 2 * math.acos(min(1.0, d[0]))))
            ax = norm([d[1], d[2], d[3]])
            rot = qmul(qfrom_angle_axis(ang, ax), sim['rot'])
        self.root_adjust(pos, rot)

    def warp_to_target(self):
        if not self.target:
            return
        a = self.anim_target()
        if not a:
            return
        p, P, dt, t = self.p, self.pose, self.dt, self.target
        e = [t['pos'][0] - a['pos'][0], t['pos'][1] - a['pos'][1], 0.0]
        yaw_err = wrap_angle(t['yaw'] - math.atan2(a['dir'][1], a['dir'][0]))
        if length(e) > p['warpMaxDist'] or abs(yaw_err) > p['warpMaxAngle']:
            return
        hip_world_vel = qmul_vec(P.rootRot, P.hipVel)
        speed = (hypot(P.rootVel[0], P.rootVel[1]) + hypot(hip_world_vel[0], hip_world_vel[1])
                 + abs(hip_world_vel[2]))
        k = damp_factor(p['warpHalflife'], dt)
        dpos = scale(e, k)
        max_len = p['warpMaxRatio'] * speed * dt
        if length(dpos) > max_len:
            dpos = scale(norm(dpos), max_len)
        dyaw = yaw_err * k
        max_yaw = p['warpMaxRatio'] * (abs(P.rootAng[2]) + speed * 1.5) * dt
        dyaw = max(-max_yaw, min(max_yaw, dyaw))
        rq = qfrom_yaw(dyaw)
        rel = sub(P.rootPos, [a['pos'][0], a['pos'][1], 0.0])
        new_pos = add(add([a['pos'][0], a['pos'][1], 0.0], qmul_vec(rq, rel)), dpos)
        self.root_adjust([new_pos[0], new_pos[1], 0.0], qnorm(qmul(rq, P.rootRot)))

    def render_state(self):
        P = self.pose
        return {'hipPos': add(P.rootPos, qmul_vec(P.rootRot, P.hipPos)), 'hipRot': qmul(P.rootRot, P.hipRot),
                'joints': P.joints, 'root': P.rootPos, 'rootRot': P.rootRot, 'contacts': P.contacts}
