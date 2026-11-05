"""Port of lmm-g1-box's web/js/sim.js and scenario.js: the boxes, the scripted sequence and the controller step
with the grasp and the release."""
import copy
import math

from .controller import Controller
from .database import MMBackend, read_json, ROBOT_JSON
from .ik import FootLock
from .mathq import hypot, lerp, qfrom_yaw, qmul_vec, qslerp_shortest_approx, qyaw, wrap_angle

INITIAL_BOXES = [{'label': 'A', 'pos': [2.0, 0.4], 'yaw': 0.3}, {'label': 'B', 'pos': [-1.8, 2.2], 'yaw': -0.6}]


def smoothstep(x):
    t = min(1.0, max(0.0, x))
    return t * t * (3 - 2 * t)


class World:
    """Boxes carry a revision that counts the moves made by commands."""

    def __init__(self):
        self.reset([])

    def reset(self, boxes):
        self.boxes = []
        self.next_id = 1
        self.pick_id = None
        self.holding_id = None
        self.place_target = None
        self.selected_id = None
        self.walk_to = None
        self.regrasp = None
        for b in boxes:
            self.add_box(b['label'], b['pos'], b['yaw'])

    def find(self, key):
        for b in self.boxes:
            if b['id'] == key or b['label'] == key:
                return b
        return None

    def floor_boxes(self):
        return [b for b in self.boxes if b['id'] != self.holding_id]

    def add_box(self, label, pos, yaw=0.0):
        used = {b['label'] for b in self.boxes}
        if not label or label in used:
            label = next((c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c not in used), str(self.next_id))
        b = {'id': self.next_id, 'label': label, 'pos': list(pos[:2]), 'yaw': yaw, 'rev': 0}
        self.next_id += 1
        self.boxes.append(b)
        return b

    def remove_box(self, key):
        b = self.find(key)
        if not b:
            return
        self.boxes = [x for x in self.boxes if x is not b]
        if self.pick_id == b['id']:
            self.pick_id = None
        if self.holding_id == b['id']:
            self.holding_id = None
            self.place_target = None
        if self.selected_id == b['id']:
            self.selected_id = None
        if self.regrasp == b['id']:
            self.regrasp = None

    def move_box(self, key, pos, yaw=None):
        b = self.find(key)
        if not b or b['id'] == self.holding_id:
            return
        b['pos'] = list(pos[:2])
        if yaw is not None:
            b['yaw'] = yaw
        b['rev'] += 1

    def set_pick(self, key):
        b = None if key is None else self.find(key)
        if self.holding_id is not None:
            return False
        self.pick_id = b['id'] if b else None
        self.walk_to = None
        return True

    def set_place(self, pos, yaw):
        if self.holding_id is None:
            return False
        self.place_target = {'pos': list(pos[:2]), 'yaw': yaw}
        self.walk_to = None
        return True

    def clear_targets(self):
        self.pick_id = None
        self.place_target = None
        self.regrasp = None

    def pick_box(self):
        return None if self.pick_id is None else self.find(self.pick_id)

    def drop(self, key, pos, yaw):
        """The physical box left the hands: it is where it was measured, and the robot picks it up again."""
        b = self.find(key)
        if not b or self.holding_id != b['id']:
            return False
        self.holding_id = None
        b['pos'] = list(pos[:2])
        b['yaw'] = yaw
        self.pick_id = b['id']
        self.regrasp = b['id']
        self.walk_to = None
        return True


def stick_from_keys(keys, yaw):
    f = (1 if 'W' in keys else 0) - (1 if 'S' in keys else 0)
    l = (1 if 'A' in keys else 0) - (1 if 'D' in keys else 0)
    if not f and not l:
        return None
    n = hypot(f, l)
    f, l = f / n, l / n
    c, s = math.cos(yaw), math.sin(yaw)
    return [c * f - s * l, s * f + c * l]


def scenario_steps():
    after = lambda s: (lambda ctx: ctx['stepTime'] >= s)
    return [
        ('Pick up box A', after(1.0), lambda w: w['setPick']('A')),
        ('Box A jumps 1.4 m while the robot bends down to it',
         lambda ctx: (ctx['phase'] == 'pick-pre' and ctx['hipz'] < 0.71) or ctx['holding'],
         lambda w: w['moveBoxBy']('A', -0.3, -1.4, 1.0)),
        ('Carry box A with the keyboard (W, then W+D)', lambda ctx: ctx['holding'] and ctx['uprightTime'] > 1.0,
         lambda w: w['drive']([[2.2, 'W'], [1.8, 'WD']])),
        ('Put box A down at the blue marker', lambda ctx: ctx['driveDone'] and ctx['stepTime'] > 0.5,
         lambda w: w['setPlaceAhead'](2.0, 0.8, 0.5)),
        ('The marker jumps while the robot carries the box to it', lambda ctx: ctx['holding'] and ctx['stepTime'] > 1.8,
         lambda w: w['setPlaceAhead'](1.6, -1.3, -0.4)),
        ('Pick up box B', lambda ctx: not ctx['holding'] and ctx['uprightTime'] > 1.0, lambda w: w['setPick']('B')),
        ('Box B jumps during the approach', after(1.6), lambda w: w['moveBoxBy']('B', 1.0, -0.7, 0.8)),
        ('Put box B down next to box A', lambda ctx: ctx['holding'] and ctx['uprightTime'] > 1.0,
         lambda w: w['setPlaceNear']('A', 0.0, 1.0, 0.0)),
        ('The marker moves to the other side of box A while box B is being lowered',
         lambda ctx: (ctx['phase'] == 'place-pre' and ctx['hipz'] < 0.71) or not ctx['holding'],
         lambda w: w['movePlaceBy'](0.0, -2.0)),
        ('End', lambda ctx: not ctx['holding'] and ctx['uprightTime'] > 1.5, lambda w: w['finish']()),
    ]


class Sim:
    STATE = ('world', 'scenario', 'sim_time', 'stats', 'force_search', 'grip', 'parked', 'held_pose', 'render_joints')

    def __init__(self, controller, foot_lock, cfg, box_size=None):
        self.controller = controller
        self.foot_lock = foot_lock
        self.cfg = cfg
        self.box_size = list(box_size or cfg['box_size'])
        self.world = World()
        self.scenario = None
        self.sim_time = 0.0
        self.stats = {'lowTime': 0.0, 'uprightTime': 0.0}
        self.force_search = False
        self.grip = None
        self.parked = None
        self.held_pose = None
        self.render_joints = None
        self.on_step = None
        self.speculative = False
        self.steps = scenario_steps()

    @classmethod
    def build(cls, loco='emmboth', **opts):
        """The sbplace variant on the given Locomotion database, as the live page starts it."""
        backend = MMBackend(loco=loco)
        robot = read_json(ROBOT_JSON)
        controller = Controller(backend.cfg, backend, robot['bodies'], **opts)
        controller.set_loco(loco)
        return cls(controller, FootLock(robot['bodies']), backend.cfg)

    def save(self):
        state = {k: copy.deepcopy(getattr(self, k)) for k in self.STATE}
        feet = copy.deepcopy((self.foot_lock.feet, self.foot_lock.skate))
        return state, feet, self.controller.save()

    def restore(self, s):
        state, feet, ctrl = s
        for k, v in state.items():
            setattr(self, k, v)
        self.foot_lock.feet, self.foot_lock.skate = feet
        self.controller.restore(ctrl)

    def check_tag(self):
        w = self.world
        target = (f'place {w.holding_id}' if w.holding_id is not None else f'pick {w.pick_id}' if w.pick_id is not None else None)
        return {'target': target, 'bending': self.controller.bent_for_event()}

    def reset_robot(self, pos=(0.0, 0.0, 0.0), yaw=0.0):
        self.controller.reset(list(pos), yaw, False)
        self.foot_lock.reset()
        self.grip = None
        self.parked = None
        self.held_pose = None
        self.render_joints = None

    def start_scripted(self):
        self.world.reset(INITIAL_BOXES)
        self.reset_robot()
        self.scenario = {'index': 0, 'stepTime': 0.0, 'drive': None, 'driveDone': False, 'done': False}
        self.sim_time = 0.0
        self.stats = {'lowTime': 0.0, 'uprightTime': 0.0}

    @property
    def done(self):
        return bool(self.scenario and self.scenario['done'])

    def relative(self, ahead, left):
        """A point `ahead` m along the robot's facing direction and `left` m to its left."""
        P = self.controller.pose
        yaw = qyaw(P.rootRot)
        return ([P.rootPos[0] + ahead * math.cos(yaw) - left * math.sin(yaw),
                 P.rootPos[1] + ahead * math.sin(yaw) + left * math.cos(yaw)], yaw)

    def scenario_api(self):
        w = self

        def set_place_ahead(ahead, left, turn=0.0):
            pos, yaw = w.relative(ahead, left)
            w.world.set_place(pos, yaw + turn)
            w.force_search = True

        def move_place_by(dx, dy, turn=0.0):
            t = w.world.place_target
            if t:
                w.world.set_place([t['pos'][0] + dx, t['pos'][1] + dy], t['yaw'] + turn)
                w.force_search = True

        def set_place_near(k, dx, dy, yaw):
            b = w.world.find(k)
            if b:
                w.world.set_place([b['pos'][0] + dx, b['pos'][1] + dy], yaw)
                w.force_search = True

        def move_box_by(k, dx, dy, turn=0.0):
            b = w.world.find(k)
            if b:
                w.world.move_box(k, [b['pos'][0] + dx, b['pos'][1] + dy], b['yaw'] + turn)
                w.force_search = True

        def drive(segments):
            w.world.clear_targets()
            w.scenario['drive'] = {'segments': segments, 'i': 0, 't': 0.0, 'yaw': qyaw(w.controller.pose.rootRot)}
            w.scenario['driveDone'] = False

        def finish():
            w.scenario['done'] = True

        return {'setPick': w.world.set_pick, 'setPlaceAhead': set_place_ahead, 'movePlaceBy': move_place_by,
                'setPlaceNear': set_place_near, 'moveBoxBy': move_box_by, 'drive': drive, 'finish': finish}

    def tick_scenario(self, dt):
        sc = self.scenario
        if not sc or sc['done']:
            return
        waiting = self.world.regrasp is not None
        if not waiting:
            sc['stepTime'] += dt
        if sc['drive']:
            d = sc['drive']
            d['t'] += dt
            if d['t'] > d['segments'][d['i']][0]:
                d['i'] += 1
                d['t'] = 0.0
            if d['i'] >= len(d['segments']):
                sc['drive'] = None
                sc['driveDone'] = True
                sc['stepTime'] = 0.0
        if waiting or sc['index'] >= len(self.steps):
            return
        title, when, act = self.steps[sc['index']]
        c = self.controller
        ctx = dict(self.stats, stepTime=sc['stepTime'], mode=c.mode, phase=c.box_phase(), driveDone=sc['driveDone'],
                   holding=self.world.holding_id is not None, hipz=c.hip_height())
        if when(ctx):
            act(self.scenario_api())
            sc['index'] += 1
            sc['stepTime'] = 0.0
            if self.on_step and not self.speculative:
                self.on_step(title, sc['index'])

    def floor_pose(self, box):
        return {'pos': [box['pos'][0], box['pos'][1], self.box_size[2] / 2], 'rot': qfrom_yaw(box['yaw'])}

    def step(self, stick=None, slow=False, face=None):
        dt, w = self.controller.dt, self.world
        if self.scenario:
            self.tick_scenario(dt)
        inp = {'stick': self.drive_stick() if self.scenario and self.scenario['drive'] else stick, 'slow': slow, 'face': face,
               'holding': w.holding_id is not None, 'pickBox': w.pick_box(), 'placeTarget': w.place_target,
               'boxes': w.floor_boxes(), 'boxHalf': [self.box_size[0] / 2, self.box_size[1] / 2],
               'forceSearch': self.force_search}
        self.force_search = False
        c = self.controller
        c.update(inp)

        if c.events['grasp'] and inp['pickBox']:
            box, turn = inp['pickBox'], 2 * math.pi / self.cfg['box_sides']
            x = qmul_vec(c.held_box_world()['rot'], [1.0, 0.0, 0.0])
            box['yaw'] = wrap_angle(box['yaw'] + turn * round_js(wrap_angle(math.atan2(x[1], x[0]) - box['yaw']) / turn))
            w.holding_id = box['id']
            w.pick_id = None
            if w.regrasp == box['id']:
                w.regrasp = None
            self.grip = {'id': box['id'], 'from': self.floor_pose(box), 't': 0.0, 'landing': False}
        if c.events['release'] and w.holding_id is not None:
            box = w.find(w.holding_id)
            hb = c.held_box_world()
            t = c.commit or c.target
            x = qmul_vec(hb['rot'], [1.0, 0.0, 0.0])
            box['pos'] = list(t['pos']) if t else [hb['pos'][0], hb['pos'][1]]
            box['yaw'] = t['yaw'] if t else math.atan2(x[1], x[0])
            self.grip = {'id': box['id'], 'from': hb, 't': 0.0, 'landing': True}
            w.holding_id = None
            w.place_target = None
        self.held_pose = self.above_floor(self.update_held(dt))
        self.render_joints = self.foot_lock.apply(c.render_state(), dt)
        self.stats['lowTime'] = self.stats['lowTime'] + dt if c.is_low() else 0.0
        self.stats['uprightTime'] = self.stats['uprightTime'] + dt if c.is_upright() else 0.0
        self.sim_time += dt

    def drive_stick(self):
        d = self.scenario['drive']
        return stick_from_keys(set(d['segments'][d['i']][1]), d['yaw'])

    def drop(self, key, pos, yaw):
        if not self.world.drop(key, pos, yaw):
            return False
        self.grip = None
        self.parked = None
        self.held_pose = None
        self.force_search = True
        return True

    def above_floor(self, pose):
        if not pose:
            return pose
        half = [s / 2 for s in self.box_size]
        depth = sum(abs(qmul_vec(pose['rot'], e)[2]) * half[i] for i, e in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])))
        lift = depth - pose['pos'][2]
        return dict(pose, pos=[pose['pos'][0], pose['pos'][1], pose['pos'][2] + lift]) if lift > 0 else pose

    def flat_on_floor(self, pose):
        x = qmul_vec(pose['rot'], [1.0, 0.0, 0.0])
        return {'pos': [pose['pos'][0], pose['pos'][1], self.box_size[2] / 2], 'rot': qfrom_yaw(math.atan2(x[1], x[0]))}

    def update_held(self, dt):
        """World pose of the box in the robot's hands, blended at the grasp and settled at the release."""
        w, c, g = self.world, self.controller, self.grip
        if g:
            g['t'] += dt
            if g['t'] > 0.3:
                self.grip = g = None
        if g and g['landing']:
            box = w.find(g['id'])
            if not box:
                return None
            to, k = self.floor_pose(box), smoothstep(g['t'] / 0.25)
            return {'id': g['id'], 'pos': lerp(g['from']['pos'], to['pos'], k), 'rot': qslerp_shortest_approx(g['from']['rot'], to['rot'], k)}
        if w.holding_id is None:
            self.parked = None
            return None
        anim = c.held_box_world()
        if c.pose.held < 0.5:
            if not self.parked:
                last = self.held_pose if self.held_pose and self.held_pose['id'] == w.holding_id else anim
                self.parked = self.flat_on_floor(last)
            return dict(self.parked, id=w.holding_id)
        if self.parked:
            self.grip = g = {'id': w.holding_id, 'from': self.parked, 't': 0.0, 'landing': False}
            self.parked = None
        if g and g['id'] == w.holding_id:
            k = smoothstep(g['t'] / 0.3)
            return {'id': g['id'], 'pos': lerp(g['from']['pos'], anim['pos'], k), 'rot': qslerp_shortest_approx(g['from']['rot'], anim['rot'], k)}
        return dict(anim, id=w.holding_id)

    def kin_qpos(self):
        """The reference pose (36): the controller's pose with locked feet, so root warping becomes steps."""
        st = self.controller.render_state()
        joints = self.render_joints if self.render_joints is not None else st['joints']
        return [*st['hipPos'], *st['hipRot'], *[float(v) for v in joints]]


def round_js(x):
    """Math.round: halves go up."""
    return math.floor(x + 0.5)
