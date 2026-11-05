"""The box task: lmm-g1-box's motion generation and tracking, streamed to the deploy node, with SLAM (the map server's
pose) and live.sam3box's boxes (the map server's objects), on the real G1 or the deploy repo's sim.

It runs the way of lmm-g1-box's page (variant holosoma-v3-box338x335x360-apv4-scenebox-sbplace-angle60, Play tab,
physical mode with the SONIC release tracker), ported in mm_g1:
  controller   exact Motion Matching over data/ (the Locomotion Matching Database 'emmboth', the page's default,
               and the Box Matching Database of sbplace), inertialization, foot locking, root warping and the
               moving-target rules
  reference    60 Hz poses resampled to 50 Hz and made 0.2 s at a time, snap-xyyaw every 10 frames onto the
               measured pelvis, held back while the box controller is bent down
  grasp        while the reference holds a box, each palm pressed 40 N toward it (tau = J^T f at the measured
               joints, ramped over 0.15 s), and both shoulder rolls opened 0.4 rad while the box controller reaches
               with empty hands, on the node's arm_tau topic
  boxes        a box reported moved is moved in the controller's world (the page's box sync), and a reference
               that lifts while the box is still seen on the floor drops the hold, so the controller picks the
               box up again where it lies (the page's grip check)

What differs from the page, because the robot is measured, not simulated:
  - the stream runs --lookahead 60 frames past each commit (the page: 25). The node plays a frame only once 46
    frames follow it, and with 25 it played 0.3 to 0.7 s behind the bridge
  - the robot is the SLAM pelvis, compared with the reference frame of the pose's age plus --track-delay 0.1 s
    (SONIC's tracking delay). The page knows the frame the robot tracked
  - a seed turns --yaw-gain 0.3 of the yaw gap (the page: all of it). SONIC tracks the reference's heading, and the
    SLAM pelvis yaw sways with the gait at another phase: full turns swung the approach by up to 23 deg
  - the box is live.sam3box's report, a few times a second and some 0.2 to 0.6 s late: a move is taken after
    --sync-need reports that agree, the grip check after --grip-need reports
  - the palm force aims at a fresh report of the box on the floor (the page: the physical box) until the
    reference lifts it, then at the box where the reference holds it, carried onto the measured pelvis, as the
    reports of a moving box lag. It turns off when a fresh report puts the box on the floor more than 0.45 m
    (horizontally) from the palms. A late report of the box on the floor is not taken as the box left behind
    while the hands lift it.
  - in locomotion no seed puts the controller's simulation object inside a box's collision zone (the runner's
    kept_out). The controller pushes it out again at once, and after a place, with the robot standing 0.2 to 0.3 m
    closer to the box than the reference, the seeds every 0.2 s made the reference shuffle and the robot kicked
    the box
  - a grip-check drop clears the place target: the drop comes some 2 s after the grasp, often after the place was
    asked for, and the controller then put down the box it had just failed to lift again

    python sonic_g1_box/run_slam.py                        # the window (UI 2, sonic_g1_box/box_window.py)
    python sonic_g1_box/run_slam.py --terminal             # terminal keys (letter + Enter)
    python sonic_g1_box/run_slam.py --headless 240 --auto-start --start-after-anchor

The window has the loco driver's keys (motionmatching-g1-loco sonic_g1_loco/mm_driver.py): ] start, P planner <->
POSE, O stop, Enter re-send the mode, W A S D move (along the controller's heading, F: camera-relative), the arrows
face, Shift runs (0.8 m/s, Shift 1.2 m/s), T walks UI 1's trajectory, Esc quits. In planner mode the same keys walk
the node's own planner, and the reference is put on the robot every 0.2 s until POSE. The box task adds X (the box
reported nearest the robot, again: the next one), a left click on a reported box (that one, in locomotion only),
B (put it down ahead), C (cancel), V (the reference's path ahead) and L (the legend). A pick or a place goes before
the trajectory, and a key held takes over from both, as on lmm-g1-box's page.

Terminal keys (type the letter, then Enter):
  ]  start (planner mode, the robot stands)     p  planner <-> POSE
  x  pick up the box reported nearest the robot (x again: the next one)
  b  put the held box down --place-ahead m in front of the robot
  w  walk forward / stop (with or without the box)
  c  cancel the pick or the place               o  stop (damping, final)       q  quit

Headless: waits for the node (and the anchor), starts, stands --settle s, takes the box reported nearest the robot,
enters POSE at --pose-after, picks it up, carries it --carry-s s forward, puts it down --place-ahead m in front,
and stands.
"""
import os

for _k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_k, '1')     # one thread searches these sizes faster than many, and leaves the CPU free

import argparse
import json
import math
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import msgpack
import numpy as np
import zmq

from deploy_link import BUS_PUB, MODE_IDLE, MODE_PLANNER, MODE_POSE, MODE_STOPPED, BusPublisher, DeployLink, RobotState
from mm_g1.mathq import hypot, qyaw
from mm_g1.runner import CONTROL_DT, GripCheck, StreamRunner, box_yaw_near, pelvis_yaw, sync_box, to_present
from mm_g1.sim import Sim
from route_follow import ROUTE_NAMES, RouteFollow

STATE_TOPIC = '/mm_driver/state'
GRID_DIM = 17 * 11
L_SHOULDER_ROLL, R_SHOULDER_ROLL = 16, 23              # in the MuJoCo order of the 29 joints
MUJOCO_TO_ISAACLAB = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20,
                               27, 21, 28])           # isaac[i] = mujoco[MUJOCO_TO_ISAACLAB[i]]
GRASP_REACH = 0.45
GRASP_RAMP = 0.15
SHOULDER_ROLL_KP = 14.251                              # the node's default shoulder roll kp (the page's SONIC kp too)
G1_XML = os.path.join(ROOT, 'assets', 'unitree_g1', 'g1.xml')
MODEL_VERSION = 'lmm-g1-box sbplace, Locomotion emmboth'
LOCO_NAMES = {0: 'IDLE', 1: 'SLOW_WALK', 2: 'WALK', 3: 'RUN'}
TASK_STATES = {'pick-pre': 'PICK', 'pick-post': 'PICK', 'place-pre': 'PLACE', 'place-post': 'PLACE'}
HUD = {'locomotion': ('Locomotion', 'Carrying'), 'navigate': ('Walking to the box', 'Carrying the box to the marker'),
       'pick-pre': ('Picking up',) * 2, 'pick-post': ('Lifting the box',) * 2, 'place-pre': ('Putting down',) * 2,
       'place-post': ('Standing up',) * 2}


def quat_yaw(q):
    return pelvis_yaw([0.0, 0.0, 0.0, *q])


class MapLink:
    """REQ to live.server (5591), reconnecting after a timeout."""

    def __init__(self, endpoint, timeout_ms=80, retry_s=0.0):
        self.endpoint, self.timeout_ms = endpoint, timeout_ms
        self.retry_s, self.down_until = retry_s, 0.0
        self.ctx = zmq.Context.instance()
        self._open()

    def _open(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.connect(self.endpoint)

    def request(self, q):
        if self.retry_s and time.monotonic() < self.down_until:
            return None
        try:
            self.sock.send(msgpack.packb(q, use_bin_type=True))
            r = msgpack.unpackb(self.sock.recv(), raw=False)
            return r if r.get('ok') else None
        except zmq.ZMQError:
            self.sock.close(0)
            self._open()
            self.down_until = time.monotonic() + self.retry_s
            return None


def report_yaw(o):
    T = np.asarray(o['T_world_object'], float).reshape(4, 4)
    return float(math.atan2(T[1, 0], T[0, 0]))


def is_active(o):
    return str(o.get('track_state', 'ACTIVE')).upper() != 'INACTIVE'


class BoxReports:
    """The reports of the box the world links to. live.sam3box starts a new track when the box jumps more than its
    --assoc-m, so the box is the newest active report of its track or within --assoc-radius of where it was last
    measured or where the world has it. A measurement is new when its stamp is (bbox_stamp_ns)."""

    def __init__(self, assoc_radius, fresh_s):
        self.assoc, self.fresh_s = assoc_radius, fresh_s
        self.track = None
        self.last = None             # {'pos': (3,), 'yaw', 'stamp', 'wall'}
        self.objects = []

    def link(self, o):
        self.track = int(o['track_id'])
        self.last = None
        self.update([o], None)

    def update(self, objects, world_xy):
        """Returns True when a new measurement of the box came in."""
        self.objects = objects
        if self.track is None:
            return False
        anchors = [a for a in (None if self.last is None else self.last['pos'][:2], world_xy) if a is not None]
        cands = [o for o in objects if is_active(o) and (int(o['track_id']) == self.track or any(
            hypot(o['bbox_center_world'][0] - a[0], o['bbox_center_world'][1] - a[1]) < self.assoc for a in anchors))]
        if not cands:
            return False
        stamp = lambda o: int(o.get('bbox_stamp_ns') or 0) or time.time_ns()       # the sim's truth has no stamp
        o = max(cands, key=stamp)
        s = stamp(o)
        if self.last is not None and s <= self.last['stamp']:
            return False
        self.track = int(o['track_id'])
        self.last = {'pos': np.asarray(o['bbox_center_world'], float), 'yaw': report_yaw(o), 'stamp': s,
                     'wall': time.time()}
        return True

    def fresh(self, within=None):
        w = self.fresh_s if within is None else within
        return self.last if self.last is not None and time.time() - self.last['wall'] < w else None


class PalmGrasp:
    """The page's grasp heuristic (engine.js graspStep) on the measured robot. Positions are in the pelvis heading
    frame: the origin under the pelvis on the floor's height, x along the heading, z up from the floor. The robot is
    posed there from the IMU's tilt and the measured joints, at the height that puts the lower sole on the floor."""

    PALM = np.array([0.10, 0.007, 0.0])        # the page's palm sites on the wrist yaw links, y mirrored on the right

    def __init__(self, force, shoulder_open, shoulder_kp, box_half_z):
        import mujoco
        self.mj = mujoco
        self.force, self.open_rad, self.kp = float(force), float(shoulder_open), float(shoulder_kp)
        self.unlifted_z = box_half_z + 0.05
        m = self.model = mujoco.MjModel.from_xml_path(G1_XML)
        self.data = mujoco.MjData(m)
        self.wrist = [m.body(f'{s}_wrist_yaw_link').id for s in ('left', 'right')]
        arm = sorted((m.jnt_qposadr[j], m.jnt_dofadr[j]) for j in range(m.njnt) if 22 <= m.jnt_qposadr[j] < 36)
        self.dof = [d for _, d in arm]                                  # the 14 arm dofs, joints 15..28
        feet = {m.body(f'{s}_ankle_roll_link').id for s in ('left', 'right')}
        self.sole = [g for g in range(m.ngeom) if m.geom_bodyid[g] in feet and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        self.jacp = np.zeros((3, m.nv))
        self.ramp = 0.0
        self.on = False
        self.palms = None

    def pose_robot(self, base_quat, body_q):
        """The palms (2, 3) in the pelvis heading frame."""
        mj, m, d = self.mj, self.model, self.data
        yaw = quat_yaw(base_quat)
        c, s = math.cos(yaw / 2), math.sin(yaw / 2)
        w, x, y, z = base_quat
        d.qpos[:] = m.qpos0
        d.qpos[0:2] = 0.0
        d.qpos[3:7] = [c * w + s * z, c * x + s * y, c * y - s * x, c * z - s * w]       # (yaw)^-1 * base_quat
        d.qpos[7:36] = body_q
        mj.mj_kinematics(m, d)
        low = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in self.sole)
        d.qpos[2] -= low
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        return [d.xpos[b] + d.xmat[b].reshape(3, 3) @ (self.PALM * [1.0, sg, 1.0]) for b, sg in zip(self.wrist, (1.0, -1.0))]

    def step(self, dt, frame, robot, held_box, floor_box):
        """frame: the reference frame the robot plays. robot: (base_quat, body_q) or None. held_box: the box where the
        reference holds it, floor_box: a fresh report of the box on the floor, both in the pelvis heading frame or
        None. Returns tau (14,) for arm_tau."""
        tau = np.zeros(14)
        if robot is None:
            self.ramp, self.on, self.palms = 0.0, False, None
            return tau
        palms = self.palms = self.pose_robot(*robot)
        mid = 0.5 * (palms[0] + palms[1])
        on = frame.held is not None
        if on and floor_box is not None:
            on = bool(hypot(floor_box[0] - mid[0], floor_box[1] - mid[1]) < GRASP_REACH)
        k = dt / GRASP_RAMP
        self.ramp = min(1.0, self.ramp + k) if on else max(0.0, self.ramp - k)
        self.on = on
        if not on and frame.open:
            tau[L_SHOULDER_ROLL - 15] += self.open_rad * self.kp
            tau[R_SHOULDER_ROLL - 15] -= self.open_rad * self.kp
        if self.force <= 0 or self.ramp <= 0:
            return tau
        # The page pulls each palm toward the physical box. A report of the box on the floor is late but still right,
        # as the box rests until the hands lift it, so it is the target until the reference's box rises. Then the
        # box moves and its reports lag, and the target is the box where the reference holds it.
        if floor_box is not None and (held_box is None or held_box[2] < self.unlifted_z):
            target = floor_box
        else:
            target = mid if held_box is None else held_box
        mj, m, d = self.mj, self.model, self.data
        for h in range(2):
            n = target - palms[h]
            L = float(np.linalg.norm(n))
            if L < 1e-6:
                continue
            mj.mj_jac(m, d, self.jacp, None, palms[h], self.wrist[h])
            arm = slice(7 * h, 7 * h + 7)
            tau[arm] += self.jacp[:, self.dof[arm]].T @ ((self.ramp * self.force / L) * n)
        return tau


def to_heading_frame(p, xy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = p[0] - xy[0], p[1] - xy[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, p[2]])


class Bridge:
    def __init__(self, args):
        self.args = args
        self.sim = Sim.build(loco=args.loco)
        self.sim.box_size = list(args.box_size)
        self.sim.reset_robot()
        step = self.sim.step
        self.sim.step = lambda: step(*self.controller_input())
        self.max_speed = self.sim.controller.p['walkFwd']      # m/s at full stick, the planner's scale too
        self.runner = None
        self.link = DeployLink(args.bind, args.port)
        self.robot = RobotState(args.robot_host, args.robot_port)
        self.map = MapLink(args.map_rep)
        self.bus = BusPublisher(args.bus_pub)
        self.reports = BoxReports(args.assoc_radius, args.fresh_s)
        self.grip = GripCheck(args.box_size[2] / 2, args.grip_phys_dz, args.grip_need)
        self.grasp = PalmGrasp(args.palm_force, args.shoulder_open, args.shoulder_kp, args.box_size[2] / 2)
        self.box_turn = 2 * math.pi / self.sim.cfg['box_sides']
        self.mode = MODE_IDLE
        self.t0 = None
        self.cur = 0
        self.published = -1
        self.pending_from = 0
        self.slam = None
        self.slam_wall = 0.0
        self.slam_at = None             # monotonic time the SLAM pose was measured at
        self.last_seed_xy = None
        self.seed_skips = 0
        self.pose_corr_yaw = 0.0        # the runner's seed yaw when POSE started
        self.last = {'heartbeat': -1.0, 'planner': -1.0, 'bus': -1.0, 'objects': -1.0, 'pose': -1.0, 'tau': None}
        self.box_id = None
        self.move_seen = None
        self.stick = (np.zeros(3), np.zeros(3))     # (move, face): the loco driver's left and right sticks
        self.last_face = np.array([1.0, 0.0, 0.0])   # map frame, the robot's heading when the planner starts
        self.planner_start_yaw = 0.0   # the robot's map yaw when this bridge last switched the node into PLANNER
        self.last_planner_cmd = None
        self.walking = False
        self.walk_dir = None
        self.route_follow = RouteFollow(self, args.tour, args.tour_speed)
        self.lost = False
        self.drops = 0
        self.syncs = 0
        self.phase = 'idle'
        self.trace = open(args.trace, 'w') if args.trace else None

    # -- the map server -----------------------------------------------------------------------------------------
    def poll_pose(self):
        r = self.map.request({'op': 'pose'})
        p = None if r is None else r.get('pose')
        if p is not None and p.get('age_s', 9.0) < 0.5:
            self.slam, self.slam_wall = p, time.time()
            self.slam_at = time.monotonic() - float(p.get('age_s', 0.0))
        elif time.time() - self.slam_wall > 0.5:
            self.slam = None
        return self.slam

    def poll_objects(self):
        r = self.map.request({'op': 'objects'})
        if r is None:
            return False
        b = self.world_box()
        return self.reports.update(r.get('objects') or [], None if b is None else b['pos'])

    def status(self):
        return self.map.request({'op': 'status'})

    def robot_xy(self):
        if self.slam is not None:
            return np.asarray(self.slam['root_xy'][:2], float)
        return np.asarray(self.sim.controller.pose.rootPos[:2], float)

    # -- the world ------------------------------------------------------------------------------------------------
    def world_box(self):
        return None if self.box_id is None else self.sim.world.find(self.box_id)

    def ordered_reports(self):
        """The reports, active tracks first, then by distance from the robot."""
        xy = self.robot_xy()
        objs = [o for o in self.reports.objects if o.get('bbox_center_world')]
        objs.sort(key=lambda o: (not is_active(o), hypot(o['bbox_center_world'][0] - xy[0], o['bbox_center_world'][1] - xy[1])))
        return objs

    def next_report(self, objs):
        """The one x takes from ordered_reports(): the first, or with a box taken the one after it, round to the first."""
        ids = [int(o['track_id']) for o in objs]
        cur = self.reports.track
        return objs[(ids.index(cur) + 1) % len(objs)] if cur in ids and self.box_id is not None else objs[0]

    def pick_nearest(self, request=True):
        """The box reported nearest the robot (active tracks first); again: the next one, round to the first."""
        if self.sim.world.holding_id is not None:
            print('[box] the robot holds the box: b puts it down', flush=True)
            return False
        self.poll_objects()
        objs = self.ordered_reports()
        if not objs:
            if request:
                print('[box] x: the map server reports no object', flush=True)
            return False
        self.take(self.next_report(objs), request)
        return True

    def select_report(self, o):
        """A click on a reported box in the window: that one is the box and the robot picks it up, in locomotion only."""
        w, c = self.sim.world, self.sim.controller
        if w.holding_id is not None:
            print('[box] click: the robot holds the box, b puts it down first', flush=True)
            return False
        if c.mode != 'loco':
            print(f'[box] click: the box controller is busy ({c.semantics}), no other box until it is done', flush=True)
            return False
        self.take(o)
        return True

    def take(self, o, request=True):
        """Report o is the box: in the controller's world in place of the one before, and the pick asked for."""
        w, xy = self.sim.world, self.robot_xy()
        c, yaw = o['bbox_center_world'], report_yaw(o)
        if self.box_id is not None:
            w.remove_box(self.box_id)
        b = w.add_box('A', [c[0], c[1]], yaw)
        self.box_id = b['id']
        self.reports.link(o)
        self.move_seen = None
        print(f"[box] box: track #{o['track_id']} at ({c[0]:.2f}, {c[1]:.2f}) m, yaw {math.degrees(yaw):.0f} deg, "
              f'{hypot(c[0] - xy[0], c[1] - xy[1]):.2f} m from the robot', flush=True)
        if request:
            self.request_pick()

    def request_pick(self):
        if self.box_id is None or not self.sim.world.set_pick(self.box_id):
            return False
        self.walking = False
        self.sim.force_search = True
        print('[box] pick requested', flush=True)
        return True

    def place_ahead(self):
        w = self.sim.world
        if w.holding_id is None:
            print('[box] b: no box in the hands', flush=True)
            return False
        self.walking = False
        pos, yaw = self.sim.relative(self.args.place_ahead, self.args.place_left)
        w.set_place(pos, yaw)
        self.sim.force_search = True
        print(f'[box] place at ({pos[0]:.2f}, {pos[1]:.2f}) m, {self.args.place_ahead:.2f} m ahead of the controller',
              flush=True)
        return True

    def cancel(self):
        self.sim.world.clear_targets()
        self.walking = False
        self.sim.force_search = True
        print('[box] targets cleared', flush=True)

    def walk(self, on):
        """The terminal's w and the headless carry: forward along the controller's heading now, until called off."""
        self.walking = on
        yaw = qyaw(self.sim.controller.pose.rootRot)
        self.walk_dir = [math.cos(yaw), math.sin(yaw)]
        print(f"[box] {'walking forward' if on else 'stop walking'}", flush=True)

    def heading(self):
        return qyaw(self.sim.controller.pose.rootRot)

    def move_stick(self):
        """(3,) the move stick in units of max_speed: the window's keys or the route, else the walk forward."""
        move = self.stick[0]
        if np.linalg.norm(move) < 1e-6 and self.walking:
            return np.array([*self.walk_dir, 0.0])
        return move

    def controller_input(self):
        """(stick, slow, face) for the controller's step, stick and face [x, y] world directions or None."""
        move, face = self.move_stick(), self.stick[1]
        return ([float(move[0]), float(move[1])] if np.linalg.norm(move) > 1e-6 else None, False,
                [float(face[0]), float(face[1])] if np.linalg.norm(face) > 1e-6 else None)

    def has_target(self):
        w = self.sim.world
        return w.pick_id is not None or w.place_target is not None

    # -- modes ---------------------------------------------------------------------------------------------------
    def start(self):
        if self.mode == MODE_STOPPED:
            print('[box] stopped (damping): restart deploy.sh to go on', flush=True)
            return
        self.link.command(start=True, planner=True)
        self.mode = MODE_PLANNER
        self.planner_started()
        print('[box] start=1 planner=1 -> PLANNER (stand)', flush=True)

    def planner_started(self):
        """The planner faces where the robot faces now, also when the node was already in planner mode with
        another +x (a bridge started again)."""
        p = self.poll_pose()
        yaw = float(p['root_yaw']) if p is not None else self.heading()
        self.planner_start_yaw = yaw
        self.last_face = np.array([math.cos(yaw), math.sin(yaw), 0.0])

    def toggle_pose(self):
        if self.mode == MODE_PLANNER:
            self.link.command(planner=False)
            self.mode = MODE_POSE
            if self.runner is not None:
                self.pose_corr_yaw = self.runner.corr[0]
                # the node plays the newest chunk from its first frame: without this one, the chunk of the last
                # replan, which starts up to 9 frames before the bridge's frame
                newest = self.runner.produced - 1
                self.send(max(self.runner.base, min(self.cur, newest)), newest)
                self.published = newest
            print('[box] planner=0 -> POSE (the controller drives the robot)', flush=True)
        elif self.mode == MODE_POSE:
            self.link.command(planner=True)
            self.mode = MODE_PLANNER
            self.planner_started()
            print('[box] planner=1 -> PLANNER (stand)', flush=True)
        else:
            print('[box] press ] first', flush=True)

    def resend_mode(self):
        if self.mode in (MODE_PLANNER, MODE_POSE):
            self.link.command(planner=self.mode == MODE_PLANNER)

    def stop(self):
        self.link.command(stop=True, planner=self.mode == MODE_PLANNER)
        self.mode = MODE_STOPPED
        print('[box] stop=1 -> damping', flush=True)

    # -- the stream ----------------------------------------------------------------------------------------------
    def begin(self):
        """The controller starts where the robot stands: the SLAM pelvis, else the map origin."""
        p = self.poll_pose()
        xy = [0.0, 0.0] if p is None else list(p['root_xy'][:2])
        yaw = 0.0 if p is None else float(p['root_yaw'])
        self.sim.reset_robot([xy[0], xy[1], 0.0], yaw)
        self.runner = StreamRunner(self.sim, lookahead=self.args.lookahead, yaw_gain=self.args.yaw_gain)
        print(f'[box] the controller starts at ({xy[0]:.2f}, {xy[1]:.2f}) m, yaw {math.degrees(yaw):.0f} deg', flush=True)

    def replan(self, F):
        """POSE: the page's snap-xyyaw, --yaw-gain of the yaw gap, against the frame the robot played when the pose
        was measured. IDLE and PLANNER: all of the yaw, against the frame of the pose's time, as the loco driver
        snaps its controller onto the robot before POSE. The robot does not play the stream then, and so POSE starts
        where the node's planner walked it."""
        robot, p, gain = None, None, None
        if self.mode != MODE_STOPPED:
            p = self.poll_pose()
        if p is not None:
            xy = np.asarray(p['root_xy'][:2], float)
            jump = 0.0 if self.last_seed_xy is None else float(np.linalg.norm(xy - self.last_seed_xy))
            self.last_seed_xy = xy
            age = float(p.get('age_s', 0.0))
            if self.mode != MODE_POSE:
                robot, gain = ((xy[0], xy[1]), float(p['root_yaw']), age), 1.0
            elif jump > self.args.max_seed_jump:
                self.seed_skips += 1
                print(f'[box] the SLAM pose jumped {jump:.2f} m in 0.2 s: no seed this time '
                      f'({self.seed_skips} so far)', flush=True)
            else:
                robot = ((xy[0], xy[1]), float(p['root_yaw']), age + self.args.track_delay)
        if self.trace is not None:
            self.trace_replan(F, p, robot)
        r = self.runner
        seeds, kept_out = r.seeds, r.kept_out
        first, _ = r.replan(F, robot, yaw_gain=gain)
        if self.trace is not None and r.seeds > seeds:
            self.trace.write(json.dumps({'seed': r.last_seed, 'F': F}) + '\n')
        if self.trace is not None and r.kept_out > kept_out:
            self.trace.write(json.dumps({'kept_out': r.kept_out, 'F': F}) + '\n')
        self.pending_from = first if self.pending_from is None else min(self.pending_from, first)

    def trace_replan(self, F, p, robot):
        """One line per replan: the pose, the frames it is compared with, and the frame whose leg joints are nearest
        the measured ones (which frame the robot is at)."""
        r = self.runner
        meas = self.robot.get()
        best = None
        if meas is not None:
            lo = max(r.base, F - 40)
            legs = np.stack([r.frame_at(f).qpos[7:19] for f in range(lo, r.produced)])   # (M, 12)
            e = np.linalg.norm(legs - meas[1][0:12], axis=1)                             # (M,)
            best = {'frame': int(lo + np.argmin(e)), 'err': float(e.min())}
        refs = {}
        for back in (0, 2, 5, 10, 15):
            f = r.frame_at(max(r.base, F - back))
            xy, yaw = to_present(f.qpos[0:2], pelvis_yaw(f.qpos), f.corr, r.corr)
            refs[back] = [round(xy[0], 4), round(xy[1], 4), round(yaw, 4)]
        c = self.sim.controller
        self.trace.write(json.dumps({
            't': time.monotonic() - self.t0, 'F': F, 'mode': self.mode,
            'pose': None if p is None else {'xy': p['root_xy'][:2], 'yaw': p['root_yaw'], 'age': p.get('age_s'),
                                            'stamp_ns': p.get('stamp_ns')},
            'seeded': robot is not None, 'hold': bool(r.hold(c)), 'refs': refs, 'match': best,
            'phase': c.box_phase() or c.mode, 'sem': c.semantics}) + '\n')

    def send(self, first, last):
        """The node takes the stream's heading at the start of POSE as the robot's and from then on turns the robot
        with every change of it (it never sees the root position). In POSE each frame's heading is therefore sent
        without the seeds' turns since then: a seed puts the controller on the robot, and the robot does not turn
        with it. Sent with them, a yaw gap at the start of POSE never closed: each seed turned the reference toward
        the robot, the robot turned as much, and the two turned on forever."""
        r = self.runner
        lo = max(r.base, first - 1)
        frames = [r.frame_at(f) for f in range(lo, last + 1)]
        q = np.stack([f.qpos for f in frames])                                     # (M, 36), M = N or N + 1
        jp = q[:, 7:36][:, MUJOCO_TO_ISAACLAB]                                     # (M, 29) IsaacLab order
        jv = np.zeros_like(jp)
        jv[1:] = (jp[1:] - jp[:-1]) / CONTROL_DT
        bq = q[:, 3:7]                                                             # (M, 4) w, x, y, z
        if self.mode == MODE_POSE:
            half = -0.5 * (np.array([f.corr[0] for f in frames]) - self.pose_corr_yaw)   # (M,)
            c, s = np.cos(half), np.sin(half)
            bq = np.stack([c * bq[:, 0] - s * bq[:, 3], c * bq[:, 1] - s * bq[:, 2],
                           c * bq[:, 2] + s * bq[:, 1], c * bq[:, 3] + s * bq[:, 0]], axis=1)   # Rz(-yaw) * q
        k = first - lo
        self.link.pose(jp[k:], jv[k:], bq[k:], np.arange(first, last + 1))

    def flush(self):
        r = self.runner
        newest = r.produced - 1
        if newest > self.published or self.pending_from is not None:
            first = max(r.base, newest - self.args.lookahead)
            if self.pending_from is not None:
                first = max(r.base, min(first, self.pending_from))
                self.pending_from = None
            self.send(first, newest)
            self.published = newest

    def tick(self, now):
        if self.t0 is None:
            self.t0 = now
            self.begin()
        self.cur = int((now - self.t0) / CONTROL_DT)
        if self.cur >= self.runner.next_replan:
            self.replan(self.cur)
        self.flush()
        if self.mode in (MODE_PLANNER, MODE_POSE) and now - self.last['heartbeat'] >= 1.0:
            self.link.command(planner=self.mode == MODE_PLANNER)
            self.last['heartbeat'] = now
        if self.mode == MODE_PLANNER and now - self.last['planner'] >= 0.1:
            self.last['planner'] = now
            self.send_planner()
        if now - self.last['pose'] >= 0.1:
            self.last['pose'] = now
            self.poll_pose()
        if now - self.last['objects'] >= self.args.objects_s:
            self.last['objects'] = now
            if self.poll_objects():
                self.box_from_report()
        if self.last['tau'] is None or now - self.last['tau'] >= CONTROL_DT:
            self.send_arm_tau(now)
        if now - self.last['bus'] >= 0.1:
            self.last['bus'] = now
            self.publish_state()

    def planner_yaw(self):
        """The map yaw of the node's planner frame +x: the robot's heading when the planner started (] or P back
        from POSE), not map +x, found from g1_debug also by a bridge started again. Without the node's heading
        state: the robot's yaw when this bridge last switched the node into PLANNER."""
        if self.slam is not None:
            yaw = self.robot.planner_frame(float(self.slam['root_yaw']), self.slam_at)
            if yaw is not None:
                return yaw
        return self.planner_start_yaw

    def send_planner(self):
        """The loco driver's: in PLANNER the move stick walks the node's own planner, the face stick turns it. The
        sticks are in the map frame, the planner's commands in its own (planner_yaw). The loco driver sends the map
        frame as it is: right only while the robot faces map +x when the planner starts. Else W walks the robot off
        its heading, and with W along the heading the robot circles."""
        vel = self.move_stick() * self.max_speed
        speed = float(np.linalg.norm(vel))
        face = self.stick[1]
        if np.linalg.norm(face) > 1e-6:
            self.last_face = face / np.linalg.norm(face)
        elif speed >= 0.05:
            self.last_face = vel / speed
        a = -self.planner_yaw()
        c, s = math.cos(a), math.sin(a)
        to_planner = lambda v: np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], 0.0])
        if speed < 0.05:
            loco, movement, sp = 0, np.zeros(3), -1.0
        else:
            movement = to_planner(vel / speed)
            loco, sp = (1 if speed < 0.8 else (2 if speed < 2.5 else 3)), speed
        self.link.planner_command(loco, movement, to_planner(self.last_face), sp)
        if self.last_planner_cmd != loco:
            print(f'[box] planner -> {LOCO_NAMES[loco]}', flush=True)
        self.last_planner_cmd = loco

    def frame_now(self):
        return self.runner.frame_at(self.cur)

    # -- the measured box against the world (runner.js syncBoxes and checkGrip) ----------------------------------
    def box_from_report(self):
        """A new measurement of the box: the grip check while the world holds it, else the box sync."""
        w, b, meas = self.sim.world, self.world_box(), self.reports.last
        if b is None or self.runner is None:
            return
        frame = self.frame_now()
        if w.holding_id == b['id']:
            self.move_seen = None
            if self.mode == MODE_POSE and self.grip.update(frame, w.holding_id, meas):
                yaw = box_yaw_near(meas['yaw'], b['yaw'], self.box_turn)
                if self.sim.drop(b['id'], [meas['pos'][0], meas['pos'][1]], yaw):
                    self.drops += 1
                    # A place target is for a held box (set_place needs one). The page's grip check comes during the
                    # rise, before any place is asked for. Here the camera sees the box left on the floor some 2 s
                    # after the grasp, often after the place was asked for, and the target kept made the controller
                    # put down the box it had just failed to lift again.
                    w.place_target = None
                    print(f"[box] grip check: the reference lifted the box but it is still on the floor at "
                          f"({meas['pos'][0]:.2f}, {meas['pos'][1]:.2f}) m: the hold is dropped, the controller picks "
                          f'it up again ({self.drops} so far)', flush=True)
            return
        self.grip.update(frame, w.holding_id, None)
        if self.runner.pending(self.cur, b['id']) or self.grasp.ramp > 0:
            return
        if meas['pos'][2] - self.args.box_size[2] / 2 > 0.15:
            return                                                     # off the floor: not a resting pose to take
        mv = sync_box(b, meas, self.box_turn, self.args.sync_move, self.args.sync_turn, self.args.sync_search)
        if mv is None:
            self.move_seen = None
            return
        pos, yaw, search = mv
        ms = self.move_seen
        if ms is not None and hypot(pos[0] - ms['pos'][0], pos[1] - ms['pos'][1]) < self.args.sync_agree:
            ms['n'] += 1
            ms['pos'], ms['yaw'] = pos, yaw
        else:
            ms = self.move_seen = {'pos': pos, 'yaw': yaw, 'n': 1}
        if ms['n'] < self.args.sync_need:
            return
        moved = hypot(pos[0] - b['pos'][0], pos[1] - b['pos'][1])
        b['pos'], b['yaw'] = pos, yaw
        self.move_seen = None
        if search:
            self.sim.force_search = True
        self.syncs += 1
        if moved > 0.1:
            print(f'[box] the box moved {moved:.2f} m to ({pos[0]:.2f}, {pos[1]:.2f}) m: moved in the world too',
                  flush=True)

    # -- the grasp ----------------------------------------------------------------------------------------------
    def send_arm_tau(self, now):
        dt = CONTROL_DT if self.last['tau'] is None else min(0.1, now - self.last['tau'])
        self.last['tau'] = now
        if self.runner is None or self.mode != MODE_POSE:
            self.grasp.ramp = 0.0
            self.link.arm_tau(np.zeros(14))
            return
        frame = self.frame_now()
        held = floor = None
        if frame.ghost_box is not None:
            held = to_heading_frame(frame.ghost_box[0:3], frame.qpos[0:2], pelvis_yaw(frame.qpos))
        meas = self.reports.fresh(self.args.reach_fresh_s)
        if meas is not None and self.slam is not None and meas['pos'][2] - self.args.box_size[2] / 2 < self.args.grip_phys_dz:
            floor = to_heading_frame(meas['pos'], self.slam['root_xy'], float(self.slam['root_yaw']))
        self.link.arm_tau(self.grasp.step(dt, frame, self.robot.get(), held, floor))

    # -- the state for UI 1, UI 2, the recorder and box_eval ------------------------------------------------------
    def task_words(self, frame):
        """(task state, the page's HUD text, semantics) at the frame the robot plays, or the controller's without one."""
        c = self.sim.controller
        step = frame.tag['step'] if frame is not None else c.mode
        sem = frame.tag['semantics'] if frame is not None else c.semantics
        holding = frame is not None and frame.held is not None
        return TASK_STATES.get(step, 'CARRY' if holding else 'LOCOMOTION'), HUD.get(sem, (sem, sem))[1 if holding else 0], sem

    def publish_state(self):
        w, r = self.sim.world, self.runner
        cur = None if r is None else self.frame_now()
        b = self.world_box()
        task = None
        if b is not None:
            holding = cur is not None and cur.held is not None
            state, hud, sem = self.task_words(cur)
            t = w.place_target
            task = {'state': state, 'phase': self.phase, 'hud': hud, 'semantics': sem,
                    'box_xy': [round(b['pos'][0], 3), round(b['pos'][1], 3)], 'box_yaw': b['yaw'], 'box_held': holding,
                    'fixed': True, 'source': 'sam3', 'size': list(self.args.box_size), 'lost': self.lost,
                    'regrasps': self.drops, 'syncs': self.syncs,
                    'place_xy': None if t is None else [round(t['pos'][0], 3), round(t['pos'][1], 3)],
                    'grasp': {'on': self.grasp.on, 'ramp': round(self.grasp.ramp, 2)}}
        last = None if r is None else r.last_seed
        self.bus.publish(STATE_TOPIC, {
            'mode': self.mode, 'variant': 'box', 'model_version': MODEL_VERSION,
            'root_xy': None if cur is None else cur.qpos[0:2].tolist(),
            'root_yaw': None if cur is None else pelvis_yaw(cur.qpos),
            'ahead': [] if r is None else [r.frame_at(f).qpos[0:2].round(3).tolist() for f in range(self.cur, r.produced, 5)],
            'grid': [0.0] * GRID_DIM, 'grid_source': None, 'scan_xy': None, 'scan_yaw': None, 'scan_encoding': 'none',
            'map_ok': False, 'localized': self.slam is not None,
            'sync_err_m': None if last is None else last['d'],
            'heading_err_deg': None if last is None else math.degrees(last['dyaw']),
            'stick': self.move_stick()[:2].tolist(), 'max_speed': self.max_speed, 'route': self.route_follow.state(),
            'seeds': None if r is None else r.seeds, 'held_back': None if r is None else r.held_back,
            'kept_out': None if r is None else r.kept_out,
            'phase': self.phase, 'task': task})

    def close(self):
        if self.trace is not None:
            self.trace.close()
        try:
            self.link.arm_tau(np.zeros(14))
        except zmq.ZMQError:
            pass
        self.link.close()
        self.robot.close()


# ---------------------------------------------------------------------- modes

def run_headless(args):
    br = Bridge(args)
    time.sleep(1.0)                                       # PUB slow joiner
    t0 = time.monotonic()
    step, t_step, t_start, next_poll = 'wait', t0, None, 0.0
    carried = False

    def go(name, phase):
        nonlocal step, t_step
        step, t_step, br.phase = name, time.monotonic(), phase
        print(f'[headless] {phase}', flush=True)

    br.phase = 'waiting for the node' + (' and the anchor' if args.start_after_anchor else '')
    try:
        while time.monotonic() - t0 < args.headless:
            now = time.monotonic()
            poll = now >= next_poll
            if poll:
                next_poll = now + 0.5
            w, c, s = br.sim.world, br.sim.controller, br.sim
            if step == 'wait':
                if poll and args.auto_start and br.robot.get() is not None:
                    st = br.status() if args.start_after_anchor else {'anchor': {'anchored': True}}
                    if st and st['anchor']['anchored']:
                        for _ in range(3):
                            br.start()
                            time.sleep(0.1)
                        t_start = time.monotonic()
                        go('settle', 'standing (settle)')
            elif step == 'settle' and now - t_step > args.settle:
                go('find', 'waiting for a reported box')
            elif step == 'find' and poll:
                if br.pick_nearest(request=False):
                    go('stand', 'box taken, standing until POSE')
                elif now - t_step > args.fix_timeout:
                    br.lost = True
                    go('end', f'no box reported within {args.fix_timeout:.0f} s: the box was lost')
            elif step == 'stand' and now - t_start >= args.pose_after:
                br.toggle_pose()
                go('pose', 'POSE')
            elif step == 'pose' and now - t_step > args.walk_after:
                br.request_pick()
                go('pick', 'walking over and picking up')
            elif step == 'pick':
                if (w.holding_id is not None and c.mode == 'loco' and s.stats['uprightTime'] > 1.0
                        and br.frame_now().held is not None):
                    if args.carry_s > 0 and not carried:
                        carried = True
                        br.walk(True)
                        go('carry', f'carrying {args.carry_s:.0f} s forward')
                    else:
                        br.place_ahead()
                        go('place', 'putting the box down in front')
            elif step == 'carry':
                if w.holding_id is None:
                    br.walk(False)
                    go('pick', 'the box left the hands: picking it up again')
                elif now - t_step > args.carry_s:
                    br.walk(False)
                    go('carry-stop', 'stopping')
            elif step == 'carry-stop' and now - t_step > 1.5:
                br.place_ahead()
                go('place', 'putting the box down in front')
            elif step == 'place':
                if w.holding_id is None and w.pick_id is not None:
                    go('pick', 'the box left the hands: picking it up again')
                elif (w.holding_id is None and w.place_target is None and c.mode == 'loco'
                      and s.stats['uprightTime'] > 1.5 and br.frame_now().held is None):
                    go('done', 'done: standing')
                    print(f'[headless] sequence done: the box is placed, standing ({br.drops} regrasps, '
                          f'{br.syncs} box moves taken, {br.runner.seeds} seeds, {br.runner.held_back} held back, '
                          f'{br.runner.kept_out} kept out of a box)',
                          flush=True)
            br.tick(now)
            time.sleep(0.003)
        if args.auto_stop:
            br.stop()
    except KeyboardInterrupt:
        pass
    finally:
        br.close()


def run_terminal(args):
    br = Bridge(args)
    keys, lock = [], threading.Lock()

    def reader():
        for line in sys.stdin:
            with lock:
                keys.append(line.strip().lower()[:1])

    threading.Thread(target=reader, daemon=True).start()
    print('Terminal keys' + __doc__.split('Terminal keys')[1].split('Headless')[0], flush=True)
    try:
        while True:
            with lock:
                pending, keys[:] = list(keys), []
            for k in pending:
                if k == ']':
                    br.start()
                elif k == 'p':
                    br.toggle_pose()
                elif k == 'x':
                    br.pick_nearest()
                elif k == 'b':
                    br.place_ahead()
                elif k == 'w':
                    br.walk(not br.walking)
                elif k == 'c':
                    br.cancel()
                elif k == 'o':
                    br.stop()
                elif k == 'q':
                    return
            br.phase = br.sim.controller.semantics
            br.tick(time.monotonic())
            time.sleep(0.003)
    except KeyboardInterrupt:
        pass
    finally:
        br.close()


def size3(s):
    v = [float(x) for x in s.split(',')]
    if len(v) != 3 or not all(0.05 < x < 1.0 for x in v):
        raise argparse.ArgumentTypeError('three sizes in m, e.g. 0.335,0.335,0.36')
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bind', default='*')
    ap.add_argument('--port', type=int, default=5556)
    ap.add_argument('--robot-host', default='localhost')
    ap.add_argument('--robot-port', type=int, default=5557)
    ap.add_argument('--map-rep', default='tcp://127.0.0.1:5591', help="live.server's REP")
    ap.add_argument('--bus-pub', default=BUS_PUB)
    ap.add_argument('--loco', default='emmboth', choices=('nsm', 'emm', 'emmcarry', 'emmboth'),
                    help="the Locomotion Matching Database (the page's default emmboth: EMM's walks and carry spans)")
    ap.add_argument('--lookahead', type=int, default=60,
                    help='50 Hz frames made past the commit point. The node plays a frame only once 46 frames follow it '
                         '(its encoder looks 45 ahead), so 46 + the 10-frame replan step + a margin. The page uses 25: '
                         'its tracker does not wait, and here 25 left the robot 0.3-0.7 s behind the bridge')
    ap.add_argument('--box-size', type=size3, default=[0.335, 0.335, 0.36],
                    help="m, the box the controller avoids and puts down (the page's box of the variant)")
    ap.add_argument('--palm-force', type=float, default=40.0, help='N per palm toward the box (the page: 40)')
    ap.add_argument('--shoulder-open', type=float, default=0.4,
                    help='rad of outward shoulder roll while the box controller reaches with empty hands (the page: 0.4)')
    ap.add_argument('--shoulder-kp', type=float, default=SHOULDER_ROLL_KP,
                    help="the node's shoulder roll kp, which turns --shoulder-open into an arm_tau torque (the node's "
                         "default gains; 40 with the lab's --debug-arm-pd-gains)")
    ap.add_argument('--run-scale', type=float, default=1.5,
                    help="stick scale with Shift held (run), as the loco driver's: 0.8 m/s x this")
    ap.add_argument('--frame', choices=('heading', 'camera'), default='heading',
                    help="the window: W along the controller's heading or into the screen (F toggles)")
    ap.add_argument('--tour', default=None,
                    help=f"the desired trajectory to seed the map server with and walk (T): {' | '.join(ROUTE_NAMES)}, "
                         "'x,y;x,y;...' (map frame), a .json from UI 1, or 'server' (what UI 1 holds)")
    ap.add_argument('--tour-speed', type=float, default=0.8, help='m/s on the desired trajectory')
    ap.add_argument('--obstacle-radius', type=float, default=8.0,
                    help='m: the window draws the real map within this distance of the robot')
    ap.add_argument('--follow-s', type=float, default=0.3,
                    help='s: the window camera eases toward the robot with this time constant (0 = locked on it)')
    ap.add_argument('--place-ahead', type=float, default=0.8, help='m in front of the controller b puts the box down')
    ap.add_argument('--place-left', type=float, default=0.0, help='m to the left, the same way')
    ap.add_argument('--yaw-gain', type=float, default=0.3,
                    help='POSE: share of the yaw gap between the SLAM pelvis and the reference a seed turns the '
                         'controller (the page: 1). The SLAM pelvis yaw sways with the gait at another phase. The robot '
                         'does not turn with a seed (see send)')
    ap.add_argument('--track-delay', type=float, default=0.1,
                    help='s the robot plays behind the frame sent, added to the pose age when the seed looks back for the '
                         'frame to compare (SONIC\'s tracking delay: the leg joints match the frames 4-6 back in the sim)')
    ap.add_argument('--max-seed-jump', type=float, default=0.5,
                    help='m: a SLAM pose this far from the one 0.2 s before is not seeded from')
    ap.add_argument('--objects-s', type=float, default=0.1, help='s between the map server object polls')
    ap.add_argument('--fresh-s', type=float, default=1.0, help='s a box report stays fresh')
    ap.add_argument('--reach-fresh-s', type=float, default=0.5,
                    help='s a report of the box on the floor turns the palm force off when the palms are away from it')
    ap.add_argument('--assoc-radius', type=float, default=1.0,
                    help="m: a new track this near the box's last report or its place in the world is the box")
    ap.add_argument('--sync-move', type=float, default=0.03, help='m a reported box moves the world box (the page: 0.02)')
    ap.add_argument('--sync-turn', type=float, default=0.1, help='rad, the same for the turn (the page: 0.05)')
    ap.add_argument('--sync-search', type=float, default=0.1, help='m: a move this large searches at once (the page: 0.1)')
    ap.add_argument('--sync-need', type=int, default=2, help='new reports in a row that agree on a move')
    ap.add_argument('--sync-agree', type=float, default=0.06, help='m two reports of a move agree within')
    ap.add_argument('--grip-phys-dz', type=float, default=0.06,
                    help='m: a reported box centre this close to its resting height is on the floor (the page: 0.04)')
    ap.add_argument('--grip-need', type=int, default=2, help='new reports in a row for the grip check')
    ap.add_argument('--fix-range-bias', type=float, default=0.0,
                    help="accepted and not used (the deploy launcher passes it with Boxer's objects)")
    ap.add_argument('--trace', default=None, metavar='JSONL', help='write one line per replan and seed here')
    ap.add_argument('--headless', type=float, default=None, metavar='SECONDS')
    ap.add_argument('--auto-start', action='store_true', help='headless: ] once the node is up (moves the robot!)')
    ap.add_argument('--start-after-anchor', action='store_true', help='headless: ] only once the map server is anchored')
    ap.add_argument('--settle', type=float, default=5.0, help='headless: s standing before the box is taken')
    ap.add_argument('--fix-timeout', type=float, default=90.0, help='headless: s to wait for a reported box')
    ap.add_argument('--pose-after', type=float, default=12.0, help='headless: s after start, P -> POSE')
    ap.add_argument('--walk-after', type=float, default=3.0, help='headless: s in POSE before the pick')
    ap.add_argument('--carry-s', type=float, default=0.0, help='headless: s walking forward with the box before the place')
    ap.add_argument('--auto-stop', action='store_true')
    ap.add_argument('--terminal', action='store_true', help='terminal keys (letter + Enter) instead of the window')
    ap.add_argument('--smoke-frames', type=int, default=None, help='(testing) close the window after N drawn frames')
    args = ap.parse_args()
    if args.headless is not None:
        run_headless(args)
    elif args.terminal:
        run_terminal(args)
    else:
        from box_window import run_window
        run_window(Bridge(args), args)


if __name__ == '__main__':
    main()
