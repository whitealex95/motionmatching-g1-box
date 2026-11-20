"""Port of the Controller of lmm-g1-box's web/js/mm/controller.js: Locomotion with its arrive state, the
box controller, inertialization, the simulation object and trajectory, root warping and the moving-target rules.

The box controller has four states, set by its tag and by whether the robot holds the box:
  pick, empty      walk in and bend down to the box. The held output of the clip rising is the grasp.
  pick, holding    rise with the box.
  place, holding   walk in and lower the box. The held output falling is the release.
  place, empty     rise without the box.
"""
import copy
import itertools
import math

import numpy as np

from .database import NDOF, NPOSE_FEATURES, TRAJ_VEL
from .mathq import (add, hypot, damp_factor, decay_spring_quat, decay_spring_scalar, decay_spring_vec, js_sign1, length,
                    norm, qabs, qangle_between, qfrom_angle_axis, qfrom_scaled_angle_axis, qfrom_yaw, qid, qinv,
                    qinv_mul, qinv_mul_vec, qmul, qmul_inv, qmul_vec, qnorm, qslerp_shortest_approx,
                    qto_scaled_angle_axis, qyaw, scale, simple_spring_damper_quat, sub, wrap_angle)

DEFAULTS = dict(
    searchTime=0.1, inertializeHalflife=0.1, simVelocityHalflife=0.27, simRotationHalflife=0.27,
    simMaxTurnRate=60 * math.pi / 180,      # rad/s the simulation object turns at most, None: no limit
    walkFwd=0.8, walkSide=0.45, walkBack=0.4, slowScale=0.5,
    adjustPosHalflife=0.1, adjustRotHalflife=0.2, adjustPosMaxRatio=0.5, adjustRotMaxRatio=0.5,
    clampMaxDistance=0.15, clampMaxAngle=0.5 * math.pi,
    boxEnter={'pick': 1.5, 'place': 1.5}, boxLeave={'pick': 2.0, 'place': 2.0}, forceEngageRadius=0.6,
    takeOverRadius=1.0, sideMargin={'pick': 0.25, 'place': 0.25},
    boxDb='same', loco='nsm', engageRadius=3.0, boxQueryRadius=3.0,
    warpHalflife=0.2, warpMaxRatio=0.8, warpMaxDist=0.7, warpMaxAngle=1.2,
    pickPlaceSpeed=1.0, pickPlaceSlow='clip', warp='always', leaveMoved=None, reapproachTime=1.0,
    boxEntry='walk', entryLead=30, walkOn='step', bendOut='cancel', rise='search',
    locoTarget='arrive', stanceSwitch=0.05, axisLead=1.0, axisLook=0.5, arriveReach=0.75,
    transitionCost=0.0,
    zLow=0.66, zUpright=0.72, uprightTilt=None,
    graspOn=0.6, releaseOff=0.4, graspReach=0.25, floorBoxLift=0.15,
    movedTolerance=0.12, movedAngle=0.35, collisionRadius=0.36,
    standOff=0.36, approachClearance=0.3, approachFallback=0.6,
    boxAhead=0.0,
)
THRESHOLDS = ('boxEnter', 'boxLeave', 'sideMargin')
NO_EDIT = (0.0, 0.0, 0.0)
WEDGE_IN = 15 * math.pi / 180
GROUP_MIDDLE = 0.1
FSM_HISTORY = 24
QUERY_DB = {'loco': 'loco', 'arrive': 'loco', 'pick': 'box', 'place': 'box'}
ARRIVE_GROUPS = ('traj_pos', 'traj_dir')
ENGAGE_IDS = itertools.count(1)       # each take-over its own id, across the re-plans' saved and restored states


def is_box_query(kind):
    return QUERY_DB[kind] == 'box'


def query_weights(cfg, kind):
    """The default weights of a query: Locomotion without the trajectory velocity, the arrive state's own."""
    c = cfg['controllers'][QUERY_DB[kind]]
    vel = c['groups'].index(TRAJ_VEL['group']) if TRAJ_VEL['group'] in c['groups'] else -1
    if kind != 'arrive':
        return [0.0 if g == vel else w for g, w in enumerate(c['weights'])]
    out = [c['weights'][c['groups'].index(g)] for g in ARRIVE_GROUPS for _ in range(3)]
    return out + ([TRAJ_VEL['arrive']] if vel >= 0 else [])


def fill_weights(cfg, kind, w):
    """Locomotion or arrive weights given without the trajectory velocity get its default."""
    d = query_weights(cfg, kind)
    return [*w, d[-1]] if len(w) == len(d) - 1 and not is_box_query(kind) else list(w)


def turned(d, a):
    return [math.cos(a) * d[0] - math.sin(a) * d[1], math.sin(a) * d[0] + math.cos(a) * d[1]]


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


def variant_defaults(cfg):
    return {'warp': cfg.get('warp', DEFAULTS['warp']), 'leaveMoved': cfg.get('leave_moved_m', DEFAULTS['leaveMoved']),
            'rise': cfg.get('rise', DEFAULTS['rise']),
            'placeEngageRadius': cfg.get('place_engage_radius', DEFAULTS['forceEngageRadius'])}


class Controller:
    STATE = ('mode', 'boxTag', 'holding', 'sim', 'desiredRot', 'desiredVel', 'searchTimer', 'forceSearch',
             'risesFrom', 'riseWalkIn', 'commit', 'target', 'navSide', 'stance', 'nav', 'boxCost', 'lastCosts', 'matchAt',
             'searchAt', 'finishing', 'calledOff', 'engaged', 'reapproach', 'engageCheck', 'engagedBy', 'leaveBox',
             'missed', 'walkIn', 'arriving', 'semantics', 'clock', 'why', 'lastEvent', 'activity', 'goalDist', 'fsm',
             'events', 'trajectory', 'curr', 'pose', 'off', 'tsrc', 'tdst', 'navigating', 'goalBoxId', 'boxEdit')

    def __init__(self, cfg, backend, bodies, **opts):
        self.cfg = cfg
        self.p = copy.deepcopy(DEFAULTS)
        self.p.update(variant_defaults(cfg))
        self.p.update(opts)
        self.p['weights'], self.p['gains'] = {}, {}
        if 'uprightTilt' not in opts:
            self.p['uprightTilt'] = cfg.get('upright_tilt_deg')
        self.torso = torso_chain(bodies)
        self.p['floorBoxZ'] = cfg['box_size'][2] / 2 + self.p['floorBoxLift']
        self.boxEdit = {'pick': NO_EDIT, 'place': NO_EDIT}
        self.p['placeEntryFrames'] = cfg.get('place_entry_frames', 0)
        self.dt = 1.0 / cfg['fps']
        for kind in QUERY_DB:
            self.set_weights(kind, query_weights(cfg, kind))
        self.backend = backend
        self.stances = backend.scenebot_stances()
        self.navigating = False
        self.goalBoxId = None
        self.lastMatch = {}
        self.reset([0.0, 0.0, 0.0], 0.0, False)

    def set_weights(self, kind, weights):
        """The search weighs each feature group's squared error by (w / w0)^2, w0 the database's normalization weight."""
        self.p['weights'][kind] = list(weights)
        if kind == 'arrive':
            self.p['gains']['arrive'] = self.arrive_gains()
            return
        c = self.cfg['controllers'][QUERY_DB[kind]]
        gain = np.zeros(c['nfeatures'])  # (nfeatures,)
        i = 0
        for g, n in enumerate(c['group_sizes']):
            gain[i:i + n] = (weights[g] / c['weights'][g]) ** 2
            i += n
        self.p['gains'][kind] = gain
        if kind == 'loco' and 'arrive' in self.p['weights']:
            self.p['gains']['arrive'] = self.arrive_gains()

    def arrive_gains(self):
        """The Locomotion gains with the arrive weights on the x and y features of each trajectory point."""
        c, w = self.cfg['controllers']['loco'], self.p['weights']['arrive']
        gain = self.p['gains']['loco'].copy()  # (nfeatures,)
        start = 0
        for k, g in enumerate(c['groups']):
            if g in ARRIVE_GROUPS:
                j = ARRIVE_GROUPS.index(g)
                for h in range(3):
                    gain[start + 2 * h:start + 2 * h + 2] = (w[3 * j + h] / c['weights'][k]) ** 2
            if g == TRAJ_VEL['group']:
                gain[start:start + c['group_sizes'][k]] = (w[3 * len(ARRIVE_GROUPS)] / c['weights'][k]) ** 2
            start += c['group_sizes'][k]
        return gain

    def set_threshold(self, name, act, value):
        self.p[name] = dict(self.p[name], **{act: value})

    def thresholds(self, act):
        p = self.p
        return {'enter': p['boxEnter'][act], 'leave': p['boxLeave'][act], 'margin': p['sideMargin'][act]}

    def reset(self, pos, yaw, holding=False):
        rot = qfrom_yaw(yaw)
        self.mode = 'loco'
        self.boxTag = None
        self.risesFrom = None
        self.holding = holding
        self.sim = {'pos': list(pos), 'vel': [0.0, 0.0, 0.0], 'acc': [0.0, 0.0, 0.0], 'rot': list(rot), 'ang': [0.0, 0.0, 0.0]}
        self.desiredRot = list(rot)
        self.desiredVel = [0.0, 0.0, 0.0]
        self.searchTimer = 0.0
        self.forceSearch = True
        self.commit = None
        self.target = None
        self.navSide = None
        self.stance = None
        self.nav = None
        self.boxCost = math.inf
        self.lastCosts = []
        self.lastMatch = {}
        self.matchAt = {}
        self.searchAt = -math.inf
        self.finishing = False
        self.calledOff = False
        self.engaged = None
        self.reapproach = None
        self.engageCheck = None
        self.engagedBy = None
        self.leaveBox = False
        self.missed = False
        self.walkIn = False
        self.riseWalkIn = False
        self.arriving = False
        self.semantics = 'none'
        # For display only (trace, status): nothing the controller does reads them.
        self.clock = 0.0
        self.why = None
        self.lastEvent = None
        self.activity = None
        self.goalDist = None
        self.fsm = {'state': None, 'since': 0.0, 'history': []}
        self.events = {'grasp': False, 'release': False}
        self.trajectory = {'pos': [list(pos) for _ in range(4)], 'rot': [list(rot) for _ in range(4)], 'vel': [0.0, 0.0, 0.0]}
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
        qy[NPOSE_FEATURES + 14:] = 0
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
        s['pos'] = self.collide(pos, obstacles, prev)

    def in_obstacle(self, p, o):
        """How deep p is in obstacle o grown by the collision radius: depth <= 0 outside."""
        r, c, s = self.p['collisionRadius'], math.cos(o['yaw']), math.sin(o['yaw'])
        dx, dy = p[0] - o['pos'][0], p[1] - o['pos'][1]
        lx, ly, hx, hy = c * dx + s * dy, -s * dx + c * dy, o['half'][0] + r, o['half'][1] + r
        return min(hx - abs(lx), hy - abs(ly))

    def collide(self, pos, obstacles, frm=None):
        """frm: the position before the move. A robot in or on two obstacles at once moves without the part of the move
        that heads more than WEDGE_IN into either, toward its centre (pushed out of one into the other, it could not)."""
        wedged = [o for o in obstacles if self.in_obstacle(frm, o) > -1e-6] if frm is not None else []
        if len(wedged) >= 2:
            d = [pos[0] - frm[0], pos[1] - frm[1]]
            tol = math.sin(WEDGE_IN) * hypot(d[0], d[1])

            def away(o):
                u = [frm[0] - o['pos'][0], frm[1] - o['pos'][1]]
                l = hypot(u[0], u[1]) or 1.0
                return [u[0] / l, u[1] / l]

            for o in wedged:
                n = away(o)
                k = d[0] * n[0] + d[1] * n[1] + tol
                if k < 0:
                    d = [d[0] - k * n[0], d[1] - k * n[1]]
            if any(d[0] * n[0] + d[1] * n[1] + tol < -1e-9 for n in map(away, wedged)):
                return list(frm)
            return [frm[0] + d[0], frm[1] + d[1], pos[2]]
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

    def axis_point(self, pos, yaw, frm):
        """The point Locomotion heads for on the way to pos, reached facing yaw: on the line through pos along yaw,
        axisLook ahead of where frm projects onto it, at most axisLead out from pos and never past it."""
        c, s = math.cos(yaw), math.sin(yaw)
        along = c * (frm[0] - pos[0]) + s * (frm[1] - pos[1])
        k = max(-self.p['axisLead'], min(0.0, along + self.p['axisLook']))
        return [pos[0] + c * k, pos[1] + s * k]

    def axis_walk(self):
        return bool(self.stances) and self.p['locoTarget'] != 'centre'

    def approach(self, nav, obstacles):
        """(stick, face, arrive) for Locomotion walking to nav {pos, yaw} (yaw None: no facing). arrive is the pose the
        query reaches at its last trajectory point, or None."""
        p, frm = self.p, self.sim['pos']
        togo = hypot(nav['pos'][0] - frm[0], nav['pos'][1] - frm[1])
        if nav['yaw'] is not None and p['locoTarget'] == 'arrive' and togo < p['arriveReach']:
            face = [math.cos(nav['yaw']), math.sin(nav['yaw'])]
            if togo <= 0.05:
                return None, face, nav
            mag = min(1.0, togo / (60 * self.dt * p['walkFwd']))
            dr = self.steer(frm, [(nav['pos'][0] - frm[0]) / togo, (nav['pos'][1] - frm[1]) / togo], togo, obstacles)
            return [mag * dr[0], mag * dr[1]], face, nav
        aim = nav['pos'] if nav['yaw'] is None else self.axis_point(nav['pos'], nav['yaw'], frm)
        d = [aim[0] - frm[0], aim[1] - frm[1]]
        dist = hypot(d[0], d[1])
        if dist <= 0.05:
            return None, None, None
        # A walk to a box or marker centre slows down as if to stop 0.4 m short of it, a walk to a stance does not.
        mag = min(1.0, max(0.35, (dist - 0.4 if nav['yaw'] is None else dist) / 1.0))
        dr = self.steer(frm, [d[0] / dist, d[1] / dist], dist, obstacles)
        return [mag * dr[0], mag * dr[1]], None, None

    def nav_target(self, goal, activity):
        """{pos, yaw} Locomotion walks a pick or a place to: the start of the SceneBot pick or place of the side the
        last search chose for this goal, or the goal's centre with yaw None."""
        ns, st = self.navSide, self.stances[activity] if self.axis_walk() else None
        if not st or not ns or ns['activity'] != activity or ns['id'] != goal.get('id'):
            self.stance = None
            return {'pos': goal['pos'], 'yaw': None}
        side = target_sides(goal['pos'], goal['yaw'], self.cfg['box_sides'])[ns['side']]
        d = self.data_target(side['pos'], side['dir'], activity)
        c, s = d['dir']
        dyaw = side['yaw'] if d['dir'] is side['dir'] else math.atan2(s, c)
        at = [{'pos': [d['pos'][0] + c * x - s * y, d['pos'][1] + s * x + c * y], 'yaw': dyaw + yaw} for x, y, yaw in st]
        dist = lambda t: hypot(t['pos'][0] - self.sim['pos'][0], t['pos'][1] - self.sim['pos'][1])
        prev = self.stance
        k = prev['k'] if prev and prev['side'] == ns['side'] and prev['activity'] == activity and prev['k'] < len(at) else -1
        best = 0
        for i in range(1, len(at)):
            if dist(at[i]) < dist(at[best]):
                best = i
        if k < 0 or dist(at[best]) < dist(at[k]) - self.p['stanceSwitch']:
            k = best
        self.stance = {'side': ns['side'], 'activity': activity, 'k': k}
        return at[k]

    def group_close(self, obstacles, keep=None):
        """Boxes closer together than the robot can pass between: per group its centre, radius, the direction across
        the line between its first two boxes, and its boxes. Tables and the box to pick up (keep) are in no group."""
        r = self.p['collisionRadius']
        rad = lambda o: max(o['half'][0], o['half'][1])
        boxes = [o for o in obstacles if not o.get('table') and (keep is None or o['id'] != keep)]
        root = list(range(len(boxes)))

        def find(i):
            if root[i] != i:
                root[i] = find(root[i])
            return root[i]

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if hypot(a['pos'][0] - b['pos'][0], a['pos'][1] - b['pos'][1]) - rad(a) - rad(b) < 2 * r:
                    root[find(j)] = find(i)
        groups = {}
        for i, o in enumerate(boxes):
            groups.setdefault(find(i), []).append(o)
        out = []
        for members in groups.values():
            if len(members) < 2:
                continue
            pos = [seq_mean([o['pos'][k] for o in members]) for k in (0, 1)]
            R = max(hypot(o['pos'][0] - pos[0], o['pos'][1] - pos[1]) + rad(o) for o in members)
            a, b = members[0], members[1]
            u = [b['pos'][0] - a['pos'][0], b['pos'][1] - a['pos'][1]]
            l = hypot(u[0], u[1]) or 1.0
            out.append({'pos': pos, 'R': R, 'across': [-u[1] / l, u[0] / l], 'members': members})
        return out

    def steer(self, frm, direction, goal_dist, obstacles):
        """Turn away from boxes that block the straight line to the goal, to the side they (or their group) are not on.
        A robot inside a group first leaves it across the line between its boxes."""
        groups = self.group_close(obstacles, self.goalBoxId)
        group_of = {}
        for g in groups:
            for o in g['members']:
                group_of[id(o)] = g
            rel = [frm[0] - g['pos'][0], frm[1] - g['pos'][1]]
            if hypot(rel[0], rel[1]) >= g['R']:
                continue
            n = g['across']
            own = n[0] * rel[0] + n[1] * rel[1]
            s = 1 if (own if abs(own) > GROUP_MIDDLE else n[0] * direction[0] + n[1] * direction[1]) >= 0 else -1
            return [s * n[0], s * n[1]]
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
            g = group_of.get(id(o))
            gl = direction[0] * (g['pos'][1] - frm[1]) - direction[1] * (g['pos'][0] - frm[0]) if g else lateral
            side = -1 if gl > 0 else 1
            w = 2 * (1 - abs(lateral) / clearance) * (1 - dist / 1.6)
            d = [d[0] - side * w * direction[1], d[1] + side * w * direction[0]]
        l = hypot(d[0], d[1]) or 1.0
        return [d[0] / l, d[1] / l]

    def predict_trajectory(self, stick, speed, obstacles, face=None, arrive=None):
        """face: [x, y] world direction the robot turns to instead of the stick's. arrive: {pos} the trajectory reaches
        at its last point and stops at."""
        sim, dt = self.sim, 20 * self.dt
        rots, drot, dvel = [list(sim['rot'])], [list(self.desiredRot)], [list(self.desiredVel)]
        face_rot = qfrom_yaw(math.atan2(face[1], face[0])) if face else None
        max_rate, tx, tv = self.p['simMaxTurnRate'], list(sim['rot']), list(sim['ang'])
        for i in range(1, 4):
            dv = self.desired_velocity(stick, rots[i - 1], speed)
            drot.append(face_rot if face_rot else qfrom_yaw(math.atan2(dv[1], dv[0])) if stick else drot[i - 1])
            if max_rate:
                # the limit depends on the angle left at each frame: stepped frame by frame, as the simulation object
                for _ in range(20):
                    tx, tv = simple_spring_damper_quat(tx, tv, drot[i], self.p['simRotationHalflife'], self.dt, max_rate)
                x = list(tx)
            else:
                x, _ = simple_spring_damper_quat(list(sim['rot']), list(sim['ang']), drot[i], self.p['simRotationHalflife'], i * dt)
            rots.append(x)
            dvel.append(self.desired_velocity(stick, rots[i], speed))
        pos = [list(sim['pos'])]
        s = {'pos': list(sim['pos']), 'vel': list(sim['vel']), 'acc': list(sim['acc'])}
        for i in range(1, 4):
            self.sim_position_update(s, dvel[i], self.p['simVelocityHalflife'], dt, obstacles)
            pos.append(list(s['pos']))
        if arrive:
            # A cubic from the simulation object's position and velocity to rest at arrive, its entry speed cut where
            # it would carry the cubic past arrive and back.
            D = [arrive['pos'][0] - sim['pos'][0], arrive['pos'][1] - sim['pos'][1], 0.0]
            T = 3 * dt
            room, v = 3 * hypot(D[0], D[1]) / T, hypot(sim['vel'][0], sim['vel'][1])
            v0 = scale([sim['vel'][0], sim['vel'][1], 0.0], room / v) if v > room else [sim['vel'][0], sim['vel'][1], 0.0]
            for i in range(1, 4):
                t = i / 3
                pos[i] = add(sim['pos'], add(scale(v0, T * (t - 2 * t * t + t * t * t)), scale(D, 3 * t * t - 2 * t * t * t)))
        self.trajectory = {'pos': pos, 'rot': rots, 'vel': [0.0, 0.0, 0.0] if arrive else list(s['vel'])}

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
        if len(qy) > NPOSE_FEATURES + 14:
            tv = qinv_mul_vec(P.rootRot, self.trajectory['vel'])
            qy[NPOSE_FEATURES + 14], qy[NPOSE_FEATURES + 15] = tv[0], tv[1]
        return qy

    def box_query(self, target_pos, target_dir, tag):
        qy = np.zeros(self.backend.ctrl['box']['nfeatures'], dtype=np.float32)
        qy[:NPOSE_FEATURES] = self.pose_features()
        P, d = self.pose, self.data_target(target_pos, target_dir, tag)
        tp = qinv_mul_vec(P.rootRot, [d['pos'][0] - P.rootPos[0], d['pos'][1] - P.rootPos[1], 0.0])
        r = hypot(tp[0], tp[1])
        if r > self.p['boxQueryRadius']:
            tp = scale(tp, self.p['boxQueryRadius'] / r)
        td = qinv_mul_vec(P.rootRot, [d['dir'][0], d['dir'][1], 0.0])
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

    def data_target(self, pos, d, kind):
        """The target {pos, dir} of the data's motion for a target of the world: back by the edit of the motion that
        plays (boxEdit) and by boxAhead. world_target goes the other way. dir is returned as given when not turned."""
        ex, ey, eyaw = self.boxEdit.get(kind) or NO_EDIT
        x = ex + self.p['boxAhead']
        dd = turned(d, -eyaw) if eyaw else d
        return {'pos': [pos[0] - x * dd[0] + ey * dd[1], pos[1] - x * dd[1] - ey * dd[0]], 'dir': dd}

    def world_target(self, pos, d, kind):
        ex, ey, eyaw = self.boxEdit.get(kind) or NO_EDIT
        x = ex + self.p['boxAhead']
        return {'pos': [pos[0] + x * d[0] - ey * d[1], pos[1] + x * d[1] + ey * d[0]], 'dir': turned(d, eyaw) if eyaw else d}

    def anim_target(self):
        """Where the playing animation believes the box target is, in world coordinates, as world_target moves it."""
        b = self.backend
        if b.active == 'loco':
            return None
        f = b.denormalize(b.active, b.current_features())
        P = self.pose
        d = qmul_vec(P.rootRot, [float(f[NPOSE_FEATURES + 2]), float(f[NPOSE_FEATURES + 3]), 0.0])
        at = add(P.rootPos, qmul_vec(P.rootRot, [float(f[NPOSE_FEATURES]), float(f[NPOSE_FEATURES + 1]), 0.0]))
        w = self.world_target(at, d, self.boxTag or 'pick')
        return {'pos': [w['pos'][0], w['pos'][1], at[2]], 'dir': [w['dir'][0], w['dir'][1], 0.0]}

    def held_box_world(self):
        """The held box in world coordinates, boxAhead farther along its X axis (away from the carrier)."""
        P = self.pose
        ahead = qmul_vec(P.boxRot, [self.p['boxAhead'], 0.0, 0.0])
        return {'pos': add(P.rootPos, qmul_vec(P.rootRot, add(P.boxPos, ahead))), 'rot': qnorm(qmul(P.rootRot, P.boxRot))}

    def hip_height(self):
        """The pelvis height the data has at the playing frame: an edited pick keeps the data's bent and upright."""
        return self.pose.hipPos[2] - self.backend.edit_lift()

    def is_low(self):
        return self.hip_height() < self.p['zLow']

    def is_upright(self):
        return self.hip_height() > self.p['zUpright']

    def past_last_step(self):
        b = self.backend
        return b.active == 'loco' or b.from_last_step(b.active)[b.frame] == 1

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

    def box_state(self):
        """box_phase with the part before the event split in two: 'pick-step' and 'place-step' are the steps to the
        target, and 'pick-pre' and 'place-pre' start where the clip goes down for the event (MMBackend.from_bend)."""
        ph, b = self.box_phase(), self.backend
        if ph not in ('pick-pre', 'place-pre') or b.active == 'loco' or b.from_bend(b.active)[b.frame] == 1:
            return ph
        return 'pick-step' if ph == 'pick-pre' else 'place-step'

    def walk_on_from(self):
        return self.p['walkOn'] if self.p['boxEntry'] == 'walk' else 'off'

    def hand_over(self):
        """A walk that ends in a SceneBot clip plays on in that clip, at the frame the two share (MMBackend.hand_off).
        The two have the same poses there, so the transition blends nothing unless the pick motion is edited."""
        b, ph = self.backend, self.box_phase()
        if ph not in ('pick-pre', 'place-pre') or self.walk_on_from() == 'off' or b.active == 'loco':
            return
        frame = int(b.hand_off(b.active)[b.frame])
        if frame < 0:
            return
        m = {'name': b.active, 'frame': frame}
        self.transition(b.match_pose(m, self.curr.rootPos, self.curr.rootRot, self.dt, self.clip_speed(m['name'], m['frame'])))
        b.commit(m)

    def bend_out_by(self):
        return self.p['bendOut'] if self.p['boxEntry'] == 'walk' else 'leave'

    def rise_frame(self):
        """Where the robot rises when the bend of now is cancelled: {name, frame} of MMBackend.rise_of, or None when
        the clip does not go down for a grasp or a release, has no such frame, or bendOut is 'leave'."""
        b, ph = self.backend, self.box_phase()
        if ph not in ('pick-pre', 'place-pre') or self.bend_out_by() != 'cancel' or b.active == 'loco':
            return None
        frame = int(b.rise_of(b.active)[b.frame])
        return None if frame < 0 else {'name': b.rise_db(b.active), 'frame': frame}

    def bend_plays(self):
        """A SceneBot pick or place that goes down plays with no jump of the search, and so does a walk that plays on
        in one."""
        b = self.backend
        return (self.rise_frame() is not None and not b.end_of_anim()
                and (b.scenebot_pre(b.active)[b.frame] == 1 or self.walk_on_from() != 'off'))

    def rise_from(self, at, why, walk_in=False):
        """The bend is cancelled: the tag flips, and the robot rises at `at` (rise_frame). No search in this update,
        so a robot that is upright already is in the rising state first. walk_in: back in Locomotion it walks in again,
        as after leaveMoved."""
        b = self.backend
        self.why = why
        self.risesFrom = self.boxTag
        self.boxTag = 'place' if self.boxTag == 'pick' else 'pick'
        self.transition(b.match_pose(at, self.curr.rootPos, self.curr.rootRot, self.dt, self.clip_speed(at['name'], at['frame'])))
        b.commit(at)
        self.forceSearch = self.leaveBox = self.missed = False
        self.searchTimer = self.p['searchTime'] / self.clip_speed(b.active)
        self.riseWalkIn = walk_in

    def bend_of(self):
        """The bend ('pick' or 'place') whose height and manual edit both SceneBot clips are solved for
        (Sim.sync_pick_motion): the bend that goes down, and the one that ended without its grasp or release while
        the robot rises from it with the other clip. So a cancelled bend rises at a frame with the same joints, also
        for a box on a table or a put down with palms of its own."""
        if self.bend_out_by() != 'cancel':
            return None
        ph, frm = self.box_phase(), self.risesFrom
        if ph == 'pick-pre':
            return 'pick'
        if ph == 'place-pre':
            return 'place'
        return frm if ph == ('place-post' if frm == 'pick' else 'pick-post') else None

    def call_off(self):
        self.calledOff = True
        self.forceSearch = True

    # ---- the frame update ----

    def update(self, inp):
        """inp: stick [x, y] world direction or None, face [x, y] or None, walkTo {pos, yaw} or None, slow, holding,
        pickBox {pos, yaw, id, table, sides} or None, placeTarget {pos, yaw, z, table, sides} or None, boxes (the boxes
        on the floor [{pos, yaw, id}]), boxHalf [x, y], tables [{id, pos, yaw, half, table}], forceSearch.
        face is not only the page's: like the stick, it also takes over from the box task (the loco driver's arrows)."""
        dt, p, b = self.dt, self.p, self.backend
        if self.holding and not inp['holding'] and self.mode == 'box' and self.boxTag == 'pick':
            self.boxTag = 'place'
            self.risesFrom = 'pick'
        self.holding = inp['holding']
        self.events = {'grasp': False, 'release': False}
        ss = p['slowScale'] if inp.get('slow') else 1.0
        speed = {'fwd': p['walkFwd'] * ss, 'side': p['walkSide'] * ss, 'back': p['walkBack'] * ss}
        goal = inp['placeTarget'] if inp['holding'] else inp['pickBox']
        activity = ('place' if inp['placeTarget'] else None) if inp['holding'] else ('pick' if inp['pickBox'] else None)
        self.why = None
        self.activity = activity
        self.goalDist = hypot(goal['pos'][0] - self.pose.rootPos[0], goal['pos'][1] - self.pose.rootPos[1]) if goal else None
        pick_box = inp['pickBox']
        obstacles = [{'id': o['id'], 'pos': o['pos'], 'yaw': o['yaw'], 'half': inp['boxHalf']} for o in inp['boxes']
                     if not (self.mode == 'box' and pick_box and o['id'] == pick_box['id'])] + list(inp.get('tables') or [])

        user = inp.get('stick') or inp.get('face')
        stick, face, arrive = inp.get('stick'), inp.get('face'), None
        navigating = False
        self.nav = None
        walk_to = None if goal else inp.get('walkTo')
        self.goalBoxId = pick_box['id'] if activity == 'pick' else None
        if not stick and not face and (goal or walk_to) and self.mode == 'loco':
            nav = self.nav_target(goal, activity) if goal else walk_to
            stick, face, arrive = self.approach(nav, obstacles)
            if goal:
                navigating = bool(stick or arrive)
                self.nav = nav
        self.navigating = navigating
        self.arriving = bool(arrive)
        # The stance of a pick is closer to the box than collisionRadius: on the way there the box being picked and the
        # table under the target stop neither the simulation object nor the trajectory. steer still turns round them.
        to_stance = self.nav is not None and self.nav['yaw'] is not None
        blocking = [o for o in obstacles if not (activity == 'pick' and o['id'] == pick_box['id'])
                    and not (goal.get('table') and o['id'] == goal['table'])] if to_stance else obstacles

        dvel = self.desired_velocity(stick, self.sim['rot'], speed)
        dvel_change = length(sub(dvel, self.desiredVel)) / dt
        self.desiredVel = dvel
        if face:
            self.desiredRot = qfrom_yaw(math.atan2(face[1], face[0]))
        elif stick and hypot(stick[0], stick[1]) > 0.01:
            self.desiredRot = qfrom_yaw(math.atan2(dvel[1], dvel[0]))
        self.predict_trajectory(stick, speed, blocking, face, arrive)
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

        # With a frame to rise at (rise_frame), from where the clip goes down, and also by what else ends the bend
        # without its event.
        at = self.rise_frame()
        abort = False
        if pre and (at or (not rising and not self.is_upright())) and (self.commit or self.target):
            t = self.target
            c = self.commit or {'boxId': pick_box['id'] if phase == 'pick-pre' and pick_box else None,
                                'side': t['side'], 'pos': list(t['pos']), 'yaw': t['yaw'], 'dir': list(t['dir'])}
            want = 'pick' if phase == 'pick-pre' else 'place'
            moved = lambda: hypot(sides[c['side']]['pos'][0] - c['pos'][0], sides[c['side']]['pos'][1] - c['pos'][1])
            turn = lambda: abs(wrap_angle(sides[c['side']]['yaw'] - c['yaw']))
            bent = not self.is_upright()
            cause = ('check' if self.calledOff else 'cleared' if activity != want else 'stick' if user
                     else 'other' if pick_box and phase == 'pick-pre' and c['boxId'] != pick_box['id']
                     else 'missed' if at and b.after_event(b.active)[b.frame] == 1
                     else 'moved' if bent and moved() > p['movedTolerance']
                     else 'turned' if bent and turn() > p['movedAngle']
                     else 'far' if at and self.leaveBox else None)
            abort = cause is not None
            if abort and not self.commit:
                self.commit = c
            if abort:
                self.why = {'by': f'cancel-{cause}'}
                if cause == 'moved':
                    self.why.update(moved=moved(), limit=p['movedTolerance'])
                if cause == 'turned':
                    self.why.update(turned=turn(), limit=p['movedAngle'])
                if cause == 'far':
                    self.why.update(moved=hypot(goal['pos'][0] - self.engaged['pos'][0], goal['pos'][1] - self.engaged['pos'][1]),
                                    limit=p['leaveMoved'])
            if abort and at:
                self.rise_from(at, self.why, cause in ('missed', 'far'))
            elif abort:
                self.boxTag = 'place' if self.boxTag == 'pick' else 'pick'
                self.forceSearch = True
        self.calledOff = False

        finishing = rising and not self.risen() and not b.end_of_anim()
        self.finishing = finishing
        self.hand_over()
        search_due = not finishing and (self.forceSearch or self.searchTimer <= 0 or b.end_of_anim())
        if search_due:
            self.searchAt = self.clock
            self.search(inp, sides, activity, goal, obstacles)
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
            self.lastEvent = {'kind': 'grasp', 't': self.clock}
            if not self.commit and self.target:
                self.commit = {'boxId': pick_box['id'], 'side': self.target['side'], 'pos': list(self.target['pos']),
                               'yaw': self.target['yaw'], 'dir': list(self.target['dir'])}
        place_z = (inp['placeTarget'] or {}).get('z') or 0.0
        if ph == 'place-pre' and held <= p['releaseOff'] and hb['pos'][2] < p['floorBoxZ'] + place_z:
            self.events['release'] = True
            self.lastEvent = {'kind': 'release', 't': self.clock}
            if not self.commit and self.target:
                self.commit = {'boxId': None, 'side': self.target['side'], 'pos': list(self.target['pos']),
                               'yaw': self.target['yaw'], 'dir': list(self.target['dir'])}

        if self.mode == 'loco':
            self.sim_position_update(self.sim, self.desiredVel, p['simVelocityHalflife'], dt, blocking)
            self.sim['rot'], self.sim['ang'] = simple_spring_damper_quat(self.sim['rot'], self.sim['ang'], self.desiredRot,
                                                                         p['simRotationHalflife'], dt, p['simMaxTurnRate'])
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
        self.semantics = (self.box_state() if self.mode != 'loco' else 'arrive' if self.arriving
                          else 'navigate' if navigating else 'locomotion')
        self.clock += dt
        self.trace(bool(goal))

    # ---- the record of the state machine and the numbers its rules compare (display only) ----

    def trace(self, has_goal):
        """Each change of state with its cause, noted where the decision was made (why) or told from the two states. A
        state is the semantics and what plays it: 'navigate:loco', 'navigate:carry', 'pick-step:pick', 'pick-pre:pick'."""
        f, s = self.fsm, self.semantics
        key = f'{s}:pick' if s.startswith('pick') else f'{s}:place' if s.startswith('place') else f"{s}:{'carry' if self.holding else 'loco'}"
        if key == f['state']:
            return
        if f['state'] is not None:
            f['history'].append({'t': self.clock, 'from': f['state'], 'to': key, **(self.why or self.infer_why(f['state'], key, has_goal))})
            if len(f['history']) > FSM_HISTORY:
                f['history'].pop(0)
        f['state'] = key
        f['since'] = self.clock

    def infer_why(self, frm, to, has_goal):
        fs, fk = frm.split(':')
        ts, tk = to.split(':')
        walks = lambda x: x in ('locomotion', 'navigate', 'arrive')
        togo = hypot(self.nav['pos'][0] - self.sim['pos'][0], self.nav['pos'][1] - self.sim['pos'][1]) if self.nav else None
        pre = lambda x: x.endswith('-pre') or x.endswith('-step')
        if pre(fs) and pre(ts) and fk == tk:
            b = self.backend
            w = b.walk_ranges(b.active) if b.active != 'loco' else []
            return ({'by': 'bend', 'sceneBot': any(x['a'] <= b.frame < x['b'] for x in w)} if ts.endswith('-pre')
                    else {'by': 'jump-back'})
        if fk == 'pick' and pre(fs) and ts == 'pick-post':
            return {'by': 'grasp'}
        if fk == 'place' and pre(fs) and ts == 'place-post':
            le = self.lastEvent
            return {'by': 'release'} if le and le['kind'] == 'release' and self.clock - le['t'] < 0.5 else {'by': 'drop', 'again': has_goal}
        if (fk == 'carry' and tk == 'loco') or (fs == 'pick-post' and ts == 'place-post'):
            return {'by': 'drop', 'again': has_goal}
        if walks(fs) and walks(ts):
            if fk != tk:
                return {'by': 'held'}
            if ts == 'arrive':
                return {'by': 'near', 'togo': togo, 'reach': self.p['arriveReach']}
            if ts == 'navigate':
                return {'by': 'far', 'togo': togo, 'reach': self.p['arriveReach']} if fs == 'arrive' else {'by': 'target'}
            return {'by': 'reached' if has_goal else 'cleared'}
        return {'by': 'other'}

    def play_cost(self):
        """The cost of the clip that plays against the query of now, which another frame must beat."""
        b = self.backend
        if self.fsm['state'] is None:
            return None
        if self.mode == 'loco':
            kind = 'arrive' if self.arriving else 'loco'
            query = self.loco_query(self.holding)
        else:
            ph = self.box_phase()
            t = (ph in ('pick-post', 'place-post') and self.commit) or self.target
            if not t or not self.boxTag:
                return None
            kind = self.boxTag
            query = self.box_query(t['pos'], t['dir'], kind)
        name = self.db_of(kind)
        return b.play_cost(name, b.normalize(name, query), self.p['gains'][kind])

    def status(self):
        """The numbers the rules of the state machine compare now (Controller.status of the page)."""
        p, f = self.p, self.fsm
        th = self.thresholds(self.activity or self.boxTag or 'pick')
        cost = {k: (self.lastMatch[k]['cost'] if k in self.lastMatch else None) for k in QUERY_DB}
        query = self.boxTag if self.mode != 'loco' else 'arrive' if self.arriving else 'loco'
        at = self.matchAt.get(query)
        frozen = None if at is None else 'rise' if self.finishing else 'unmatched' if at < self.searchAt else None
        return {
            'state': f['state'], 'since': self.clock - f['since'], 'last': f['history'][-1] if f['history'] else None,
            'clock': self.clock, 'mode': self.mode, 'query': query, 'activity': self.activity, 'cost': cost,
            'frozen': frozen, 'costAge': None if at is None else self.clock - at, 'play': self.play_cost(),
            'boxCost': self.boxCost if math.isfinite(self.boxCost) else None, 'sides': list(self.lastCosts),
            'side': self.target['side'] if self.target else None, 'searchedAt': dict(self.matchAt),
            'enter': th['enter'], 'leave': th['leave'], 'margin': th['margin'], 'dist': self.goalDist,
            'engage': p['placeEngageRadius'] if self.activity == 'place' else p['forceEngageRadius'],
            'limit': p['takeOverRadius'], 'search': p['engageRadius'],
            'wait': None if self.reapproach is None else max(0.0, p['reapproachTime'] - self.reapproach),
            'togo': hypot(self.nav['pos'][0] - self.sim['pos'][0], self.nav['pos'][1] - self.sim['pos'][1]) if self.nav else None,
            'reach': p['arriveReach'] if self.axis_walk() and p['locoTarget'] == 'arrive' else None,
            'hip': self.hip_height(), 'upright': p['zUpright'], 'low': p['zLow'], 'tilt': self.torso_tilt(),
            'tiltMax': p['uprightTilt'],
        }

    def approach_clear(self, frm, side, obstacles, kind='pick'):
        """A side is usable when the straight walk to where the robot stands for it (standOff behind the target, and
        boxAhead more) stays clear of the other floor boxes."""
        p, t = self.p, self.data_target(side['pos'], side['dir'], kind)
        stand = [t['pos'][0] - p['standOff'] * t['dir'][0], t['pos'][1] - p['standOff'] * t['dir'][1]]
        d = [stand[0] - frm[0], stand[1] - frm[1]]
        len2 = d[0] * d[0] + d[1] * d[1] or 1e-9
        for o in obstacles:
            u = max(0.0, min(1.0, ((o['pos'][0] - frm[0]) * d[0] + (o['pos'][1] - frm[1]) * d[1]) / len2))
            gap = hypot(frm[0] + u * d[0] - o['pos'][0], frm[1] + u * d[1] - o['pos'][1])
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
        m = self.match('arrive' if self.arriving else 'loco', qy)
        self.transition(b.match_pose(m, self.curr.rootPos, self.curr.rootRot, self.dt))
        b.commit(m)

    def db_of(self, kind):
        return kind if is_box_query(kind) and self.p['boxDb'] == 'different' else QUERY_DB[kind]

    def match(self, kind, query, entry=0):
        b, name = self.backend, self.db_of(kind)
        qn, gain = b.normalize(name, query), self.p['gains'][kind]
        sb_pre = (is_box_query(kind) and self.cfg.get('scenebot_entry_frames') and b.active == name
                  and b.scenebot_pre(name)[b.frame] == 1)
        on = self.walk_on_from()
        walk_on = is_box_query(kind) and on != 'off' and b.active == name and b.walk_on(name, on)[b.frame] == 1
        plays = is_box_query(kind) and b.active == name and self.bend_plays()
        mask = None
        if is_box_query(kind) and self.p['boxEntry'] == 'walk':
            mask = (b.every_frame(name) if plays else b.before_event_mask(name) if sb_pre or walk_on
                    else b.walk_in_mask(name) if self.walkIn else b.no_entry(name))
        m = b.match(name, qn, gain, entry=entry, mask=mask, transition_cost=self.p['transitionCost'])
        m.update(kind=kind, qn=qn, gain=gain)
        return m

    def search(self, inp, sides, activity, goal, obstacles):
        b, p, P = self.backend, self.p, self.pose
        th = self.thresholds(activity or self.boxTag or 'pick')
        enter, leave, margin = th['enter'], th['leave'], th['margin']
        self.lastCosts = []
        self.navSide = None
        box_match = None
        phase = self.box_phase()
        pick_box = inp['pickBox']
        user = inp.get('stick') or inp.get('face')

        if phase in ('pick-post', 'place-post'):
            # A clip at its end has nothing left to play, so the rise ends there too. An edit of the pick motion can
            # keep the torso tilted to the end.
            c = self.commit or self.target
            if c and not self.risen() and not b.end_of_anim():
                box_match = self.match(self.boxTag, self.box_query(c['pos'], c['dir'], self.boxTag))
        elif activity and sides:
            near = hypot(goal['pos'][0] - P.rootPos[0], goal['pos'][1] - P.rootPos[1])
            if self.mode == 'box' or near < p['engageRadius']:
                locked = self.mode == 'box' and self.target and (self.is_low() or self.commit)
                others = [o for o in obstacles if (not pick_box or o['id'] != pick_box['id']) and o['id'] != goal.get('table')]
                # A target on a table is taken from outside the table only.
                gs = goal.get('sides')
                open_ = [s for s in sides if s['side'] in gs] if gs is not None else sides
                cands = ([sides[self.target['side']]] if locked
                         else [s for s in open_ if self.approach_clear(P.rootPos, s, others, activity)])
                if not cands and near < p['approachFallback']:
                    cands = open_
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
                    self.navSide = {'side': best[0]['side'], 'activity': activity, 'id': goal.get('id')}
        self.boxCost = box_match['cost'] if box_match else math.inf
        if box_match:
            self.lastMatch[box_match['kind']] = box_match
            self.matchAt[box_match['kind']] = self.clock

        use_box = False
        engage = p['placeEngageRadius'] if activity == 'place' else p['forceEngageRadius']
        self.engageCheck = None
        if activity and sides:
            self.engageCheck = {'kind': activity, 'mode': self.mode, 'cost': self.boxCost, 'enter': enter,
                                'dist': hypot(goal['pos'][0] - P.rootPos[0], goal['pos'][1] - P.rootPos[1]),
                                'radius': engage, 'limit': p['takeOverRadius'], 'search': p['engageRadius'],
                                'reapproach': self.reapproach}
        if self.leaveBox or self.missed:
            self.why = ({'by': 'leave-missed'} if not self.leaveBox else
                        {'by': 'leave-moved', 'moved': hypot(goal['pos'][0] - self.engaged['pos'][0], goal['pos'][1] - self.engaged['pos'][1]),
                         'limit': p['leaveMoved']})
            self.reapproach = 0.0
            self.engaged = None
            self.walkIn = True
        elif self.mode == 'box':
            if phase in ('pick-post', 'place-post'):
                use_box = bool(box_match)
                if not use_box:
                    self.why = {'by': 'upright', 'hip': self.hip_height(), 'tilt': self.torso_tilt()}
                if not use_box and self.riseWalkIn:
                    self.reapproach = 0.0
                    self.walkIn = True
                if not use_box:
                    self.riseWalkIn = False
            elif not activity or user:
                use_box = not self.is_upright() and bool(box_match)
                if not use_box:
                    self.why = {'by': 'leave-cleared' if not activity else 'leave-stick'}
            else:
                dist = hypot(goal['pos'][0] - P.rootPos[0], goal['pos'][1] - P.rootPos[1])
                use_box = bool(box_match) and not (self.is_upright() and self.boxCost > leave and dist > engage + 0.3)
                at = None if use_box else self.rise_frame()
                if at:
                    return self.rise_from(at, {'by': 'cancel-cost', 'cost': self.boxCost, 'leave': leave, 'dist': dist,
                                               'radius': engage + 0.3} if box_match else {'by': 'cancel-noside'})
                if not use_box:
                    self.why = ({'by': 'leave-cost', 'cost': self.boxCost, 'leave': leave, 'dist': dist, 'radius': engage + 0.3}
                                if box_match else {'by': 'leave-nomatch'})
        elif box_match and not user:
            dist = hypot(goal['pos'][0] - P.rootPos[0], goal['pos'][1] - P.rootPos[1])
            within = p['takeOverRadius'] is None or dist < p['takeOverRadius']
            by_cost, by_dist = self.boxCost < enter, dist < engage and self.reapproach is None
            use_box = within and (by_cost or by_dist)
            if use_box:
                self.boxTag = activity
                self.reapproach = None
                self.why = {'by': 'both' if by_cost and by_dist else 'cost' if by_cost else 'distance', 'cost': self.boxCost,
                            'enter': enter, 'dist': dist, 'radius': engage, 'limit': p['takeOverRadius'],
                            'side': self.target['side'] if self.target else None}
                self.engagedBy = dict(self.engageCheck, by_cost=by_cost, by_dist=bool(by_dist), id=next(ENGAGE_IDS))

        if use_box:
            m = box_match
            self.mode = 'box'
        else:
            if self.mode == 'box':
                self.commit = None
                self.boxTag = None
                self.risesFrom = None
            self.mode = 'loco'
            if not activity:
                self.target = None
            kind = 'arrive' if self.arriving else 'loco'
            m = self.match(kind, self.loco_query(inp['holding']))
            self.lastMatch[kind] = m
            self.matchAt[kind] = self.clock
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
        """unblended: the pelvis and the joints of the playing frame without the offset of a transition, at the same
        root."""
        P, C = self.pose, self.curr
        return {'hipPos': add(P.rootPos, qmul_vec(P.rootRot, P.hipPos)), 'hipRot': qmul(P.rootRot, P.hipRot),
                'joints': P.joints, 'root': P.rootPos, 'rootRot': P.rootRot, 'contacts': P.contacts,
                'unblended': {'hipPos': add(P.rootPos, qmul_vec(P.rootRot, C.hipPos)), 'hipRot': qmul(P.rootRot, C.hipRot),
                              'joints': C.joints}}


def seq_mean(xs):
    """members.reduce((a, o) => a + o, 0) / n: the JS running sum."""
    s = 0.0
    for x in xs:
        s += x
    return s / len(xs)
