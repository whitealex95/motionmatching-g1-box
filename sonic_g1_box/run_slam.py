"""The box task: lmm-g1-box's motion generation and tracking, streamed to the deploy node, with SLAM (the map server's
pose) and live.sam3box's boxes (the map server's objects), on the real G1 or the deploy repo's sim.

It runs the way of lmm-g1-box's page (variant holosoma-v3-box338x335x360-apv4-scenebox-sbplace-angle60, Play tab,
physical mode with the SONIC release tracker: the page's defaults since its tag v3.0), ported in mm_g1:
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
  - a seed turns --yaw-gain of the yaw gap: 1, all of it, as the page, since 2026-11-06. The 0.3 before was set while
    the stream's heading still carried each seed's turn
  - the box is live.sam3box's report, a few times a second and some 0.2 to 0.6 s late: a move is taken after
    --sync-need reports that agree, the grip check after --grip-need reports
  - the palm force aims at a fresh report of the box on the floor (the page: the physical box) until the
    reference lifts it, then at the box where the reference holds it, carried onto the measured pelvis, as the
    reports of a moving box lag. It turns off when a fresh report puts the box on the floor more than 0.45 m
    (horizontally) from the palms. A late report of the box on the floor is not taken as the box left behind
    while the hands lift it.
  - in locomotion no seed puts the controller's simulation object inside a box's collision zone (--keep-out, the
    runner's kept_out). The controller pushes it out again at once, and after a place, with the robot standing 0.2
    to 0.3 m closer to the box than the reference, the seeds every 0.2 s made the reference shuffle and the robot
    kicked the box. While the robot stands that close, the reference stays where it is
  - a grip-check drop clears the place target: the drop comes some 2 s after the grasp, often after the place was
    asked for, and the controller then put down the box it had just failed to lift again
  - the page presses the palms only while the physical box is within 0.45 m of their midpoint and checks the grip
    on it at every step (engine.js graspStep, runner.js checkGrip). With no such box, measured palm sites closer
    together than --empty-gap (0.3 m) stop the palm force and the squeezes at once, before the hands meet, and drop
    the hold HANDS_EMPTY_S (0.2 s) later. Before 2026-11-07 the threshold was the box's width minus 0.08 m (0.255 m),
    the force stayed on for 0.3 s past it and the hands hit each other: the palm sites sit 5.3 cm inside the
    hands, so the gap was 0.15 m between the hands' surfaces when it crossed. After a drop the force stays off until the frames of the next re-plan play. UI 1 and
    UI 3 draw the last 12 s of the gap, the reference's gap, the threshold and the force ('grasp_trace'), and each
    drop says what dropped it (grip: the grip check, palms: the palm gap, pose: leaving POSE)
  - display only (2026-11-07): the map server's newest LiDAR scan in a vertical column at the measured palms'
    midpoint (its `column` op), 10 Hz, beside the palms' mean height. A hold disagrees when the column tops more
    than COLUMN_MARGIN below the palms, or holds no point where the LiDAR sees it at the palms' height. Its runs
    are 'between' events, and grasp_trace's 'column' and each drop's 'between_s' carry it. Nothing is dropped for it
  - after a drop the task goes through Locomotion before the pick again (2026-11-07). The page's controller, still
    in the pick's clip, grasps again on its next frame with the hands up in the air (graspReach is horizontal only,
    the clip's held output high mid-lift). Here the clip rises with empty hands (Stand up, as a pick cancelled while
    bent down), and the pick is set again only once the robot plays Locomotion and the camera has seen the box
  - leaving POSE (P, O) drops the hold and clears the targets: the node's planner has the arms then, and the
    reference went on carrying with its hands together, which the next POSE went back to
  - the task's state machine (FSM_STATES at the played frame, its signals and transitions) goes to UI 3 in the
    state's 'fsm', and the page's clip view of the clips the robot plays and the stream reaches in 'match' 'views'
  - each take-over of the box controller says whether the cost (below Enter) or the distance (within the engage
    radius) took it, in the transition, in 'match' 'engaged' and on the terminal ('engage' is the last check)

The page's Play panel settings are flags with the page's defaults (the data variant's config.json where the page takes
them from it), set live from UI 3's Settings tab:
  anchoring        --yaw-gain, --hold
  grasp heuristic  --palm-force, --palm-target, --shoulder-squeeze, --wrist-squeeze, --shoulder-open
  pick and place   --pick-speed, --pick-slow, --warp, --leave-moved, --box-entry, --rise, --transition-cost,
                   --upright-tilt, --box-db, --foot-lock
  matching         --weights-loco, --weights-pick, --weights-place, --box-enter, --box-leave, --side-margin
  by distance      --engage-radius, --force-engage-radius, --place-engage-radius, --reapproach-time (constants of the
                   page's controller, not on its panel)
The seed's own rules, none of them the page's, are live flags on the same card (SEED_FLAGS): --skip-stopped,
--skip-stale, --max-seed-jump (0: off), --keep-out, --pre-pose-full, --track-delay (0: off), --strip-seed-turns. Each
re-plan's outcome (seeded, or the rule that skipped the seed) and the counts are in the state's 'seed_rules'.
Not ported: the tracker choice, the other anchoring modes, the pick and place check, the box mass and the number of
sides.

    python sonic_g1_box/run_slam.py                        # the window (UI 2, sonic_g1_box/box_window.py)
    python sonic_g1_box/run_slam.py --terminal             # terminal keys (letter + Enter)
    python sonic_g1_box/run_slam.py --web                  # no window: UI 3 (http://localhost:8780/ui3)
    python sonic_g1_box/run_slam.py --headless 240 --auto-start --start-after-anchor

The window has the loco driver's keys (motionmatching-g1-loco sonic_g1_loco/mm_driver.py): ] start, P planner <->
POSE, O stop, Enter re-send the mode, W A S D move (along the controller's heading, F: camera-relative), the arrows
face, Shift runs (0.8 m/s, Shift 1.2 m/s), T walks UI 1's trajectory, Esc quits. In planner mode the same keys walk
the node's own planner, and the reference is put on the robot every 0.2 s until POSE. The box task adds X (the box
reported nearest the robot, again: the next one), a left click on a reported box (that one, in locomotion only),
B (put it down ahead), C (cancel), V (the reference's path ahead) and L (the legend). A click on the floor is the
page's: the held box is put down there, with empty hands the robot walks there. UI 3's 3-D view takes the same clicks
and drags the put-down marker. A pick or a place goes before the trajectory, and a key held takes over from both, as
on lmm-g1-box's page. The database page (db_page.py, web/, --db-port) shows the controller's state, the clip the
robot plays and the last matches, above the page's own database page.

Terminal keys (type the letter, then Enter):
  ]  start (planner mode, the robot stands)     p  planner <-> POSE
  x  pick up the box reported nearest the robot (x again: the next one)
  b  put the held box down --place-ahead m in front of the robot
  w  walk forward / stop (with or without the box)
  c  cancel the pick or the place               o  stop (damping, final)       q  quit

UI 3 (the map server's /ui3 page, motionmatching-g1-deploy live/web/ui3.js) sends the window's keys and the box
keys through the map server (its `operator` op, operator_link.py) in every mode but headless, and lists the flags:
LIVE_FLAGS apply at once, RESTART_FLAGS restart the bridge with its own command line and the new values (not in
POSE). The window's held keys go before UI 3's. `--web` runs without the window.

UI 1's box task (its Box task card, the map server's `box_task` op): the box to pick up (a reported track), the
put-down marker and how to carry the box there, and Go. BoxRoute walks it in POSE: the box is taken and picked
up, then put down at the marker (the controller walks there itself), or first carried along the desired trajectory
with Drive on and put down at the marker once the robot is within HANDOFF_M of it on the last segment (or at the
trajectory's end). A regrasp goes back to the pick and on from there. Stop in UI 1, C, O or P out of POSE end it,
and the result goes back to UI 1. Not in headless runs, which keep their own sequence.

Headless: waits for the node (and the anchor), starts, stands --settle s, takes the box reported nearest the robot,
enters POSE at --pose-after, picks it up, carries it --carry-s s forward, puts it down --place-ahead m in front,
and stands.
"""
import os

for _k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_k, '1')     # one thread searches these sizes faster than many, and leaves the CPU free

import argparse
import bisect
import collections
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

import operator_link as OP
from db_page import DbPage
from deploy_link import BUS_PUB, MODE_IDLE, MODE_PLANNER, MODE_POSE, MODE_STOPPED, BusPublisher, DeployLink, RobotState
from mm_g1.controller import QUERY_DB, torso_tilt
from mm_g1.mathq import hypot, qyaw
from mm_g1.runner import (CONTROL_DT, GRIP_REF_DZ, HOLDS, GripCheck, StreamRunner, box_yaw_near, pelvis_yaw, sync_box,
                          to_present)
from mm_g1.sim import Sim
from route_follow import ROUTE_NAMES, RouteFollow

STATE_TOPIC = '/mm_driver/state'
REF_TOPIC = '/mm_driver/ref'      # the reference's whole body (and the box it holds) at the frame the robot plays: UI 3
REF_HZ = 30.0
# The settings of lmm-g1-box's Play panel (app.js), each with the page's default: the controller's (its `p`), the grasp
# heuristic's (PalmGrasp) and the anchoring's (StreamRunner).
CONTROLLER_FLAGS = {'pick_speed': 'pickPlaceSpeed', 'pick_slow': 'pickPlaceSlow', 'warp': 'warp',
                    'leave_moved': 'leaveMoved', 'box_entry': 'boxEntry', 'rise': 'rise',
                    'transition_cost': 'transitionCost', 'upright_tilt': 'uprightTilt', 'box_db': 'boxDb',
                    'box_enter': 'boxEnter', 'box_leave': 'boxLeave', 'side_margin': 'sideMargin',
                    'engage_radius': 'engageRadius', 'force_engage_radius': 'forceEngageRadius',
                    'place_engage_radius': 'placeEngageRadius', 'reapproach_time': 'reapproachTime'}
WEIGHT_FLAGS = {'weights_loco': 'loco', 'weights_pick': 'pick', 'weights_place': 'place'}
GRASP_FLAGS = {'palm_force': 'force', 'palm_target': 'target', 'shoulder_squeeze': 'squeeze_rad',
               'wrist_squeeze': 'wrist_rad', 'shoulder_open': 'open_rad', 'shoulder_kp': 'kp', 'wrist_kp': 'wrist_kp'}
PAGE_FLAGS = ('hold', 'yaw_gain', 'foot_lock', *GRASP_FLAGS, *CONTROLLER_FLAGS, *WEIGHT_FLAGS)
# The seed's rules that are not the page's, in the order a re-plan checks them (UI 3's Anchoring card).
SEED_FLAGS = ('skip_stopped', 'skip_stale', 'max_seed_jump', 'keep_out', 'pre_pose_full', 'track_delay', 'strip_seed_turns')
LIVE_FLAGS = ('run_scale', 'tour_speed', 'frame', 'place_ahead', 'place_left', *PAGE_FLAGS, 'empty_gap', *SEED_FLAGS,
              'objects_s', 'reach_fresh_s', 'sync_move', 'sync_turn', 'sync_search', 'sync_need',
              'sync_agree', 'follow_s', 'obstacle_radius')
RESTART_FLAGS = ('loco', 'lookahead', 'box_size', 'assoc_radius', 'fresh_s', 'grip_phys_dz', 'grip_need',
                 'fix_range_bias')
GRID_DIM = 17 * 11
L_SHOULDER_ROLL, R_SHOULDER_ROLL = 16, 23              # in the MuJoCo order of the 29 joints
L_WRIST_YAW, R_WRIST_YAW = 21, 28
MUJOCO_TO_ISAACLAB = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20,
                               27, 21, 28])           # isaac[i] = mujoco[MUJOCO_TO_ISAACLAB[i]]
GRASP_REACH = 0.45
GRASP_RAMP = 0.15
EMPTY_GAP = 0.3            # m between the palm sites: the box held is ~0.44, the hands touch at ~0.11
HANDS_EMPTY_S = 0.2        # s from the force cut to the drop of the hold
GRASP_TRACE_S = 12.0       # s of palm gaps kept for UI 1 and UI 3
COLUMN_S = 0.1             # s between the map server's `column` requests (the LiDAR's 10 Hz)
COLUMN_R = 0.06            # m around the palm midpoint
COLUMN_MARGIN = 0.05       # m: the scan's top between the palms this far below the palms disagrees with a hold
COLUMN_AGREE_S = 0.3       # s of agreement that end a disagreement
FSM_STATES = ('locomotion', 'to_box', 'pick-pre', 'pick-post', 'carry', 'to_place', 'place-pre', 'place-post', 'lost')
FSM_HISTORY = 16
CLIP_FPS = 60
CLIP_VIEW_POINTS = 160                                 # samples of a clip's pelvis and torso curves sent to UI 3
WALK_DONE_M = 0.25         # app.js playStick: a walk to a clicked point ends this near it, slows down within 1 m
SHOULDER_ROLL_KP = 14.251                              # the node's default shoulder roll kp (the page's SONIC kp too)
WRIST_YAW_KP = 16.778                                  # the node's default wrist yaw kp (policy_parameters.hpp STIFFNESS_4010)
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

    def __init__(self, box_half_z, force=40.0, target='box', squeeze_rad=0.0, wrist_rad=0.0, open_rad=0.4,
                 kp=SHOULDER_ROLL_KP, wrist_kp=WRIST_YAW_KP):
        import mujoco
        self.mj = mujoco
        self.force, self.target, self.open_rad, self.kp = float(force), target, float(open_rad), float(kp)
        self.squeeze_rad, self.wrist_rad, self.wrist_kp = float(squeeze_rad), float(wrist_rad), float(wrist_kp)
        self.unlifted_z = box_half_z + 0.05
        m = self.model = mujoco.MjModel.from_xml_path(G1_XML)
        self.data = mujoco.MjData(m)
        self.wrist = [m.body(f'{s}_wrist_yaw_link').id for s in ('left', 'right')]
        arm = sorted((m.jnt_qposadr[j], m.jnt_dofadr[j]) for j in range(m.njnt) if 22 <= m.jnt_qposadr[j] < 36)
        self.dof = [d for _, d in arm]                                  # the 14 arm dofs, joints 15..28
        feet = {m.body(f'{s}_ankle_roll_link').id for s in ('left', 'right')}
        self.sole = [g for g in range(m.ngeom) if m.geom_bodyid[g] in feet and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        self.ref_data = mujoco.MjData(m)
        self.jacp = np.zeros((3, m.nv))
        self.ramp = 0.0
        self.on = False
        self.palms = None
        self.gap = None            # m between the palm sites
        self.ref_gap = None        # m between the reference's palm sites at the frame the robot plays
        self.released = False      # the hold was dropped: no force while the frames made before it play
        self.cut = False           # the palm gap fell below --empty-gap during this hold: no force until it ends

    def pose_robot(self, base_quat, body_q, d=None):
        """The palms (2, 3) in the pelvis heading frame."""
        mj, m = self.mj, self.model
        d = self.data if d is None else d
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

    def measure(self, robot, frame):
        """The measured palms and their gap (robot: (base_quat, body_q) or None), and the reference's gap at frame.
        self.data keeps the measured robot for step's Jacobians."""
        if robot is None:
            self.palms, self.gap = None, None
        else:
            self.palms = self.pose_robot(*robot)
            self.gap = float(np.linalg.norm(self.palms[0] - self.palms[1]))
        if frame is None:
            self.ref_gap = None
        else:
            p = self.pose_robot(frame.qpos[3:7], frame.qpos[7:36], self.ref_data)
            self.ref_gap = float(np.linalg.norm(p[0] - p[1]))

    def step(self, dt, frame, held_box, floor_box):
        """After measure(). frame: the reference frame the robot plays. held_box: the box where the reference holds
        it, floor_box: a fresh report of the box on the floor, both in the pelvis heading frame or None. Returns tau
        (14,) for arm_tau."""
        tau = np.zeros(14)
        if self.palms is None:
            self.ramp, self.on = 0.0, False
            return tau
        palms = self.palms
        mid = 0.5 * (palms[0] + palms[1])
        on = frame.held is not None and not self.released and not self.cut
        if on and floor_box is not None:
            on = bool(hypot(floor_box[0] - mid[0], floor_box[1] - mid[1]) < GRASP_REACH)
        k = dt / GRASP_RAMP
        self.ramp = min(1.0, self.ramp + k) if on else max(0.0, self.ramp - k)
        self.on = on
        if on:
            tau[L_SHOULDER_ROLL - 15] -= self.squeeze_rad * self.kp
            tau[R_SHOULDER_ROLL - 15] += self.squeeze_rad * self.kp
            tau[L_WRIST_YAW - 15] -= self.wrist_rad * self.wrist_kp
            tau[R_WRIST_YAW - 15] += self.wrist_rad * self.wrist_kp
        elif frame.open:
            tau[L_SHOULDER_ROLL - 15] += self.open_rad * self.kp
            tau[R_SHOULDER_ROLL - 15] -= self.open_rad * self.kp
        if self.force <= 0 or self.ramp <= 0:
            return tau
        # The page pulls each palm toward the physical box. A report of the box on the floor is late but still right,
        # as the box rests until the hands lift it, so it is the target until the reference's box rises. Then the
        # box moves and its reports lag, and the target is the box where the reference holds it.
        if self.target == 'mid':
            target = mid
        elif floor_box is not None and (held_box is None or held_box[2] < self.unlifted_z):
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


class ClipTable:
    """The clip of a row of the controller's Animation Database: the variant's clips, then the EMM takes' and the carry
    spans' in use, in the order of the page's StatePanel (states.js clipRows). view(i) is what the StatePanel draws
    of clip i, for UI 3."""

    def __init__(self, backend, torso):
        parts = [(backend.cfg['clips'], 0)] + [(e.cfg['clips'], e.first) for e in (backend.emm, backend.emm_carry) if e]
        self.clips = [(c['names'][n], bool(mirror), first + start, first + stop, attach, release)
                      for c, first in parts for n, mirror, start, stop, attach, release in c['rows']]
        self.starts = [c[2] for c in self.clips]
        self.b, self.torso = backend, torso
        self.lanes = db_lanes(backend)                                                    # (2, nframes)
        self.views = {}

    def of(self, g):
        if g is None:
            return None
        i = max(0, bisect.bisect_right(self.starts, g) - 1)
        name, mirror, start, stop = self.clips[i][:4]
        return {'i': i, 'f': int(g - start), 'name': name, 'mirror': mirror, 'at_s': round((g - start) / CLIP_FPS, 2),
                'len_s': round((stop - start) / CLIP_FPS, 2)}

    def view(self, i):
        if i not in self.views:
            self.views[i] = self.make_view(i)
        return self.views[i]

    def make_view(self, i):
        """The pelvis height and torso lean every k frames (and at the last frame), the frames of the Locomotion and
        Box Matching Databases as [start, stop, value] runs (db_lanes), the last steps and the grasp and release."""
        b, L = self.b, self.b.L
        name, mirror, start, stop, attach, release = self.clips[i]
        n = stop - start
        k = max(1, math.ceil(n / CLIP_VIEW_POINTS))
        fs = list(range(0, n, k)) + ([n - 1] if (n - 1) % k else [])
        rows = b.pose[start:stop]                                                           # (n, W)
        h, j0 = L['hip_quat'][0], L['joints'][0]
        pelvis = [round(float(rows[f, L['hip_pos'][0] + 2]), 3) for f in fs]
        lean = [round(torso_tilt(self.torso, rows[f, h:h + 4], rows[f, j0:]), 1) for f in fs]

        def runs(lane):
            v = lane[start:stop]                                                            # (n,)
            edges = [0, *(np.flatnonzero(np.diff(v)) + 1).tolist(), n]
            return [[a, e, int(v[a])] for a, e in zip(edges[:-1], edges[1:]) if v[a]]

        steps = b.steps[i] if i < len(b.steps) else []
        event = lambda f: int(f) if f is not None and 0 <= f < n else None
        return {'i': i, 'name': name, 'mirror': mirror, 'n': n, 'fps': CLIP_FPS, 'k': k, 'pelvis': pelvis, 'lean': lean,
                'loco': runs(self.lanes[0]), 'box': runs(self.lanes[1]),
                'steps': [[s['step'], s['lift'], s['kind'], s['event']] for s in steps],
                'attach': event(attach), 'release': event(release)}


def db_lanes(b):
    """markFrames of the page's StatePanel, per frame of the Animation Database: in the Locomotion Matching Database
    1 empty-handed or 2 carrying, in the Box Matching Database 1 pick or 2 place, plus 2 for the frames the box search
    does not jump into (MMBackend.noEntry), 0 in none."""
    lanes = np.zeros((2, len(b.pose)), np.uint8)                                            # (2, nframes)
    held = b.pose[:, b.L['held'][0]] > 0.5                                                  # (nframes,)
    idx = b.db['loco'].idx
    lanes[0, idx] = np.where(held[idx], 2, 1)
    d, ne = b.db['box'], b.no_entry('box')
    for r, (a, e) in enumerate(d.ranges):
        lanes[1, d.idx[a:e]] = (2 if d.range_kind[r] else 1) + 2 * ne[a:e]
    return lanes


class BoxRoute:
    """UI 1's box task, walked in POSE (the module docstring). phase: None (no task yet), wait_pose, find, to_box,
    carry, place, then done | failed | stopped with the result sent back to the map server."""

    POLL_S = 0.25
    HANDOFF_M = 1.5               # m from the marker on the last segment: the carry hands over to the place
    FIND_S = 10.0                 # s to wait for the chosen track to be reported
    ACTIVE = ('wait_pose', 'find', 'to_box', 'carry', 'place')

    def __init__(self, br):
        self.br = br
        self.seq, self.task, self.phase, self.text = None, None, None, ''
        self.t_poll, self.t_phase = -1.0, 0.0
        self.carried = False

    def active(self):
        return self.phase in self.ACTIVE

    def state(self):
        return None if self.phase is None else {'phase': self.phase, 'text': self.text, 'go_seq': self.seq,
                                                'track_id': None if self.task is None else self.task['track_id']}

    def go(self, phase, text):
        self.phase, self.text, self.t_phase = phase, text, time.monotonic()
        self.br.say(f'box task: {text}')

    def finish(self, phase, result, report=True):
        rf = self.br.route_follow
        if self.task is not None and self.task['carry'] == 'route' and rf.drive:
            rf.set_drive(False)
        self.go(phase, result)
        if report:
            self.br.map.request({'op': 'command', 'cmd': {'op': 'set_box_task', 'go': False, 'result': result}})

    def poll(self, now):
        if now - self.t_poll < self.POLL_S:
            return
        self.t_poll = now
        r = self.br.map.request({'op': 'box_task'})
        if r is None:
            return
        if r['go'] and r['go_seq'] != self.seq:
            self.seq, self.task, self.carried = r['go_seq'], r, False
            if self.br.world_box() is not None and self.br.sim.world.holding_id is not None \
                    and self.br.reports.track != r['track_id']:
                self.finish('failed', f"refused: the robot holds another box (track #{self.br.reports.track}): "
                                      'B puts it down first')
                return
            self.go('find' if self.br.mode == MODE_POSE else 'wait_pose',
                    f"box #{r['track_id']}: " + ('finding it' if self.br.mode == MODE_POSE else 'waiting for POSE (P)'))
        elif not r['go'] and self.active():
            self.br.cancel(from_task=True)
            self.finish('stopped', 'stopped from UI 1: the robot stands', report=False)

    def picked(self):
        b, w, c = self.br, self.br.sim.world, self.br.sim.controller
        return (w.holding_id is not None and c.mode == 'loco' and self.br.sim.stats['uprightTime'] > 1.0
                and b.frame_now().held is not None)

    def regrasp(self):
        w = self.br.sim.world
        return w.holding_id is None and (w.pick_id is not None or w.regrasp is not None)

    def after_pick(self):
        t, b = self.task, self.br
        if t['place'] is None:
            self.finish('done', 'done: the robot holds the box (no put-down marker)')
        elif t['carry'] == 'route' and not self.carried:
            rf = b.route_follow
            rf.t_poll = 0.0
            rf.poll(time.monotonic())
            f = rf.follower
            if f is None:
                self.carried = True
                return self.after_pick()
            if f.done:
                f.idx, f.done, rf.done = 0, False, False
            rf.set_drive(True)
            self.go('carry', f'carrying the box along the desired trajectory ({len(f.wp)} waypoints)')
        else:
            x, y, yaw = t['place']
            if b.place_at(x, y, yaw):
                self.go('place', f'putting the box down at ({x:+.2f}, {y:+.2f}) m')
            else:
                self.finish('failed', 'failed: no box in the hands for the put-down')

    def step(self, now):
        self.poll(now)
        if not self.active():
            return
        b, w = self.br, self.br.sim.world
        if b.mode == MODE_STOPPED:
            return self.finish('stopped', 'stopped: the robot is damped (O)')
        if self.phase != 'wait_pose' and b.mode != MODE_POSE:
            b.cancel(from_task=True)
            return self.finish('stopped', 'stopped: planner mode (P)')
        if self.phase == 'wait_pose':
            if b.mode == MODE_POSE:
                self.go('find', f"box #{self.task['track_id']}: finding it")
        elif self.phase == 'find':
            tid = int(self.task['track_id'])
            if w.holding_id is not None and b.reports.track == tid:
                return self.after_pick()
            o = next((o for o in b.reports.objects if o.get('bbox_center_world') and int(o['track_id']) == tid), None)
            if o is not None:
                b.take(o)
                self.go('to_box', f'walking to box #{tid} and picking it up')
            elif now - self.t_phase > self.FIND_S:
                self.finish('failed', f'failed: box #{tid} is not reported (for {self.FIND_S:.0f} s)')
        elif self.phase == 'to_box':
            if self.picked():
                self.after_pick()
            elif w.pick_id is None and w.holding_id is None and w.regrasp is None:
                self.finish('failed', 'failed: the pick was given up')
            elif (w.pick_id is None and w.regrasp is not None and b.regrasp_wait == 'seen' and b.last_drop is not None
                  and time.time() - b.last_drop['wall'] > self.FIND_S):
                self.finish('failed', f'failed: the box left the hands and is not seen on the floor '
                                      f'(for {self.FIND_S:.0f} s)')
        elif self.phase == 'carry':
            rf = b.route_follow
            f = rf.follower
            if self.regrasp():
                rf.set_drive(False)
                return self.go('to_box', 'the box left the hands: picking it up again')
            if w.holding_id is None:
                return self.finish('failed', 'failed: the box is not in the hands')
            x, y, _ = self.task['place']
            xy = b.robot_xy()
            near = f is not None and f.idx == len(f.wp) - 1 and math.hypot(x - xy[0], y - xy[1]) < self.HANDOFF_M
            if f is None or rf.done or near:
                self.carried = True
                if rf.drive:
                    rf.set_drive(False)
                self.after_pick()
            elif not rf.drive:
                self.text = 'carrying: Drive is off (T, or Drive in UI 1, walks on)'
        elif self.phase == 'place':
            if self.regrasp():
                return self.go('to_box', 'the box left the hands: picking it up again')
            if w.holding_id is not None and w.place_target is None:
                return self.finish('stopped', 'stopped: the put-down was cancelled')
            if (w.holding_id is None and w.place_target is None and b.sim.controller.mode == 'loco'
                    and b.sim.stats['uprightTime'] > 1.5 and b.frame_now().held is None):
                x, y, _ = self.task['place']
                self.finish('done', f'done: the box is down at ({x:+.2f}, {y:+.2f}) m')


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
        self.grasp = PalmGrasp(args.box_size[2] / 2, **{k: getattr(args, n) for n, k in GRASP_FLAGS.items()})
        c = self.sim.controller
        self.variant_defaults = {**{n: page_choice(n, c.p[k]) for n, k in CONTROLLER_FLAGS.items()},
                                 **{n: list(c.p['weights'][k]) for n, k in WEIGHT_FLAGS.items()}}
        self.apply_page_settings()
        self.box_turn = 2 * math.pi / self.sim.cfg['box_sides']
        self.mode = MODE_IDLE
        self.t0 = None
        self.cur = 0
        self.published = -1
        self.pending_from = 0
        self.slam = None
        self.slam_wall = 0.0
        self.slam_at = None             # monotonic time the SLAM pose was measured at
        self.raw_pose = None            # the map server's newest pose, however old
        self.last_seed_xy = None
        self.skips = {'stopped': 0, 'stale': 0, 'jump': 0}
        self.seed_outcome = None        # the last re-plan: 'seeded' or the rule that skipped its seed
        self.pose_corr_yaw = 0.0        # the runner's seed yaw when POSE started
        self.last = {'heartbeat': -1.0, 'planner': -1.0, 'bus': -1.0, 'objects': -1.0, 'pose': -1.0, 'tau': None,
                     'ref': -1.0, 'column': -1.0}
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
        self.hands_empty_s = 0.0        # s since the palm gap cut the force in this hold
        self.grasp_trace = collections.deque(maxlen=int(GRASP_TRACE_S / CONTROL_DT))
        self.grasp_events = collections.deque(maxlen=16)   # (monotonic s, 'cut' | 'grip' | 'palms' | 'pose', text)
        self.last_drop = None           # {'wall', 'by', 'why', 'between_s'}
        # the scan between the palms, display only: (monotonic s of the scan, top, points, palm z, hold on, disagrees,
        # the lowest height of the column the LiDAR sees)
        self.column_trace = collections.deque(maxlen=int(GRASP_TRACE_S / COLUMN_S) + 10)
        self.column_stamp = None
        self.column_why = None
        self.between_since = None       # monotonic s of the scan that first disagreed in this run of disagreements
        self.between_ok_since = None
        self.between_runs = 0
        self.regrasp_wait = None        # after a drop: 'seen' (until the camera sees the box on the floor), 'loco'
        self.release = None        # {'F': first re-plan's frame after the drop, 'regrasp'}: frames up to F still hold it
        self.fsm = {'state': None, 'since': time.time(), 'history': collections.deque(maxlen=FSM_HISTORY), 'why': None,
                    'engaged': None}
        self.phase = 'idle'
        self.trace = open(args.trace, 'w') if args.trace else None
        self.ui = None                  # 'window' | 'terminal' | 'web' | 'headless'
        self.flags = None               # OP.Flags: UI 3's list of the bridge's flags
        self.keys_from = None
        self.box_route = BoxRoute(self)
        self.clips = ClipTable(self.sim.controller.backend, self.sim.controller.torso)
        self.db_page = None
        if args.db_port and args.headless is None:
            try:
                self.db_page = DbPage(args.db_port)
                self.say(f'the database page: http://localhost:{args.db_port}/')
            except OSError as e:
                self.say(f'no database page: port {args.db_port}: {e}')

    # -- the map server -----------------------------------------------------------------------------------------
    def poll_pose(self):
        r = self.map.request({'op': 'pose'})
        p = None if r is None else r.get('pose')
        if r is not None:
            self.raw_pose = None if p is None else (p, time.monotonic())
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
            self.say('the robot holds the box: b puts it down')
            return False
        self.poll_objects()
        objs = self.ordered_reports()
        if not objs:
            if request:
                self.say('x: the map server reports no object')
            return False
        self.take(self.next_report(objs), request)
        return True

    def select_report(self, o):
        """A click on a reported box in the window: that one is the box and the robot picks it up, in locomotion only."""
        w, c = self.sim.world, self.sim.controller
        if w.holding_id is not None:
            self.say('click: the robot holds the box, b puts it down first')
            return False
        if c.mode != 'loco':
            self.say(f'click: the box controller is busy ({c.semantics}), no other box until it is done')
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
        self.say(f"box: track #{o['track_id']} at ({c[0]:.2f}, {c[1]:.2f}) m, yaw {math.degrees(yaw):.0f} deg, "
              f'{hypot(c[0] - xy[0], c[1] - xy[1]):.2f} m from the robot')
        if request:
            self.request_pick()

    def request_pick(self):
        if self.box_id is None or not self.sim.world.set_pick(self.box_id):
            return False
        self.walking = False
        self.sim.force_search = True
        self.say('pick requested')
        return True

    def place_ahead(self):
        w = self.sim.world
        if w.holding_id is None:
            self.say('b: no box in the hands')
            return False
        self.walking = False
        pos, yaw = self.sim.relative(self.args.place_ahead, self.args.place_left)
        w.set_place(pos, yaw)
        self.sim.force_search = True
        self.say(f'place at ({pos[0]:.2f}, {pos[1]:.2f}) m, {self.args.place_ahead:.2f} m ahead of the controller')
        return True

    def place_at(self, x, y, yaw=None, what='the put-down marker', quiet=False):
        """UI 1's put-down marker, a click on the floor or the marker dragged. No yaw: the box faces away from the
        controller, as the page's click puts it."""
        w = self.sim.world
        if w.holding_id is None:
            return False
        if yaw is None:
            P = self.sim.controller.pose.rootPos
            yaw = math.atan2(y - P[1], x - P[0])
        self.walking = False
        w.set_place([x, y], yaw)
        self.sim.force_search = True
        if not quiet:
            self.say(f'place at ({x:.2f}, {y:.2f}) m, {what}')
        return True

    def floor_click(self, x, y):
        """The page's click on the floor (app.js): with the box held, put it down there. With empty hands, clear the
        targets and walk there. It ends UI 1's box task."""
        w = self.sim.world
        if self.box_route is not None and self.box_route.active():
            self.box_route.finish('stopped', 'a click on the floor took over')
        if w.holding_id is not None:
            self.fsm_cause('a click on the floor: put the box down there')
            return self.place_at(x, y, None, 'the clicked point')
        if w.pick_id is not None or w.regrasp is not None:
            self.fsm_cause('a click on the floor: the pick is cancelled')
        w.clear_targets()
        self.walking = False
        w.walk_to = [x, y]
        self.sim.force_search = True
        self.say(f'walk to ({x:.2f}, {y:.2f}) m, the clicked point')
        return True

    def walk_stick(self):
        """(3,) app.js playStick toward the clicked point: full stick down to 1 m, then slower (0.3 at least). From the
        controller in POSE (the robot follows it), from the robot in planner mode."""
        w = self.sim.world
        P = self.sim.controller.pose.rootPos if self.mode == MODE_POSE else self.robot_xy()
        d = np.array([w.walk_to[0] - P[0], w.walk_to[1] - P[1]])
        dist = float(np.linalg.norm(d))
        if dist < WALK_DONE_M:
            w.walk_to = None
            self.say('walk: the clicked point is reached')
            return np.zeros(3)
        mag = min(1.0, max(0.3, dist))
        return np.array([mag * d[0] / dist, mag * d[1] / dist, 0.0])

    def cancel(self, from_task=False):
        self.sim.world.clear_targets()
        self.regrasp_wait = None
        self.sim.world.walk_to = None
        self.walking = False
        self.sim.force_search = True
        self.say('targets cleared')
        if not from_task:
            self.fsm_cause('C: the pick and the put-down are cancelled')
        if not from_task and self.box_route is not None and self.box_route.active():
            self.box_route.finish('stopped', 'cancelled (C)')

    def say(self, msg):
        if self.flags is None:
            print(f'[box] {msg}', flush=True)
        else:
            self.flags.say('[box]', msg)

    # -- UI 3 -------------------------------------------------------------------------------------------------
    def drive_sticks(self, now, op, local=None):
        """The sticks of one loop step: the window's held keys (local: (left, right)) go first, then UI 3's, then UI 1's
        trajectory while Drive is on and no pick or place is asked for. True when the trajectory steers."""
        args, rf = self.args, self.route_follow
        keys = None if op is None else op.keys()
        if local is not None and (np.linalg.norm(local[0]) > 1e-6 or np.linalg.norm(local[1]) > 1e-6):
            left, right = local
            self.keys_from = 'the window'
        elif keys is not None:
            az = math.degrees(self.heading()) if args.frame == 'heading' else op.cam_az_deg
            left, right = OP.keys_stick(keys, 'left', args.run_scale, az), OP.keys_stick(keys, 'right', 1.0, az)
            self.keys_from = 'UI 3'
        else:
            left, right = np.zeros(3), np.zeros(3)
            self.keys_from = None
        w = self.sim.world
        if self.keys_from is not None:
            w.walk_to = None
        elif w.walk_to is not None and self.mode in (MODE_PLANNER, MODE_POSE):
            left = self.walk_stick()
            if w.walk_to is not None:
                self.keys_from = 'the clicked point'
        rf.poll(now)
        auto = None
        if self.mode in (MODE_PLANNER, MODE_POSE) and not self.has_target() and self.keys_from is None:
            auto = rf.stick(rf.robot_xy(), self.max_speed)
            if auto is not None:
                self.keys_from = 'the desired trajectory'
        self.stick = (left, right) if auto is None else auto
        return auto is not None

    def set_flag(self, name, value):
        fl = self.flags
        if name not in fl.live:
            return self.say(f'UI 3: {name} is not a live flag')
        try:
            v = fl.value(name, value)
        except ValueError as e:
            return self.say(f'UI 3: {e}')
        if name in WEIGHT_FLAGS:
            n = len(self.sim.controller.p['weights'][WEIGHT_FLAGS[name]])
            if len(v) != n:
                return self.say(f'UI 3: {fl.flag(name)} takes {n} weights, not {len(v)}')
        setattr(self.args, name, v)
        if name == 'tour_speed':
            self.route_follow.speed = v
        elif name == 'keep_out' and self.runner is not None:
            self.runner.keep_out = v == 'on'
        elif name in PAGE_FLAGS:
            self.apply_page_settings(name)
        fl.note_live(name, v)
        self.say(f'UI 3: {fl.flag(name)} {fl.text(v)}')

    def apply_page_settings(self, only=None):
        """The page's settings from args onto the controller, the grasp heuristic, the anchoring and the foot locking.
        A setting left at None takes the controller's value (the variant's, config.json), and args shows it then."""
        a, c = self.args, self.sim.controller
        for name, key in CONTROLLER_FLAGS.items():
            if only not in (None, name):
                continue
            v = getattr(a, name)
            if v is None:
                setattr(a, name, page_choice(name, c.p[key]))
            else:
                c.p[key] = page_value(name, v)
        for name, kind in WEIGHT_FLAGS.items():
            if only not in (None, name):
                continue
            if getattr(a, name) is None:
                setattr(a, name, list(c.p['weights'][kind]))
            else:
                c.set_weights(kind, getattr(a, name))
        for name, key in GRASP_FLAGS.items():
            if only in (None, name):
                setattr(self.grasp, key, getattr(a, name) if name == 'palm_target' else float(getattr(a, name)))
        if only in (None, 'foot_lock'):
            self.sim.foot_lock.enabled = a.foot_lock == 'on'
        if self.runner is not None and only in (None, 'hold', 'yaw_gain'):
            self.runner.hold = HOLDS[a.hold]
            self.runner.yaw_gain = a.yaw_gain
            self.runner.keep_out = a.keep_out == 'on'
        if only is not None:
            self.sim.force_search = True

    def restart(self, changes, close):
        if self.mode == MODE_POSE:
            return self.say('UI 3: no restart in POSE (the robot tracks this bridge): P back to planner mode first')
        fl = self.flags
        try:
            vals = {n: fl.value(n, v) for n, v in changes.items() if n in fl.actions}
        except ValueError as e:
            return self.say(f'UI 3: restart refused: {e}')
        self.say('UI 3: restarting with ' + ' '.join(f'{fl.flag(n)} {fl.text(v)}' for n, v in vals.items()))
        self.publish_state()
        fl.restart(vals, close)

    def operator_command(self, c, close):
        k = c.get('key')
        if k == 'start':
            self.start()
        elif k == 'pose':
            self.toggle_pose()
        elif k == 'stop':
            self.stop()
        elif k == 'resend':
            self.resend_mode()
        elif k == 'pick':
            self.pick_nearest()
        elif k == 'select':
            self.poll_objects()
            o = next((o for o in self.reports.objects if int(o['track_id']) == int(c.get('track_id', -1))), None)
            if o is None:
                self.say(f"UI 3: track #{c.get('track_id')} is not reported")
            else:
                self.select_report(o)
        elif k == 'place':
            at = c.get('at')
            if at is None:
                self.place_ahead()
            elif not self.place_at(float(at[0]), float(at[1]), None if c.get('yaw') is None else float(c['yaw']),
                                   'the dragged marker', quiet=bool(c.get('drag'))) and not c.get('drag'):
                self.say('UI 3: no box in the hands to put down')
        elif k == 'floor':
            at = c.get('at')
            self.floor_click(float(at[0]), float(at[1]))
        elif k == 'cancel':
            self.cancel()
        elif k == 'set':
            self.set_flag(c.get('name'), c.get('value'))
        elif k == 'restart':
            self.restart(c.get('set') or {}, close)

    def walk(self, on):
        """The terminal's w and the headless carry: forward along the controller's heading now, until called off."""
        self.walking = on
        yaw = qyaw(self.sim.controller.pose.rootRot)
        self.walk_dir = [math.cos(yaw), math.sin(yaw)]
        self.say(f"{'walking forward' if on else 'stop walking'}")

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
            self.say('stopped (damping): restart deploy.sh to go on')
            return
        self.link.command(start=True, planner=True)
        self.mode = MODE_PLANNER
        self.planner_started()
        self.say('start=1 planner=1 -> PLANNER (stand)')

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
            self.say('planner=0 -> POSE (the controller drives the robot)')
        elif self.mode == MODE_POSE:
            self.link.command(planner=True)
            self.mode = MODE_PLANNER
            self.planner_started()
            self.say('planner=1 -> PLANNER (stand)')
            self.leave_pose('planner mode (P)')
        else:
            self.say('press ] first')

    def resend_mode(self):
        if self.mode in (MODE_PLANNER, MODE_POSE):
            self.link.command(planner=self.mode == MODE_PLANNER)

    def stop(self):
        self.link.command(stop=True, planner=self.mode == MODE_PLANNER)
        self.mode = MODE_STOPPED
        self.say('stop=1 -> damping')
        self.leave_pose('stop (O)')

    # -- the stream ----------------------------------------------------------------------------------------------
    def begin(self):
        """The controller starts where the robot stands: the SLAM pelvis, else the map origin."""
        p = self.poll_pose()
        xy = [0.0, 0.0] if p is None else list(p['root_xy'][:2])
        yaw = 0.0 if p is None else float(p['root_yaw'])
        self.sim.reset_robot([xy[0], xy[1], 0.0], yaw)
        self.runner = StreamRunner(self.sim, lookahead=self.args.lookahead, hold=self.args.hold, yaw_gain=self.args.yaw_gain,
                                   keep_out=self.args.keep_out == 'on')
        self.say(f'the controller starts at ({xy[0]:.2f}, {xy[1]:.2f}) m, yaw {math.degrees(yaw):.0f} deg')

    def seed_pose(self):
        """The pose a seed starts from, or (None, the rule that gave none): --skip-stopped, --skip-stale."""
        a = self.args
        if self.mode == MODE_STOPPED and a.skip_stopped == 'on':
            return None, 'stopped'
        p = self.poll_pose()
        if p is None and a.skip_stale == 'off' and self.raw_pose is not None:
            raw, t = self.raw_pose
            p = {**raw, 'age_s': float(raw.get('age_s', 0.0)) + time.monotonic() - t}
        return p, None if p is not None else 'stale'

    def replan(self, F):
        """POSE: the page's snap-xyyaw, --yaw-gain of the yaw gap, against the frame the robot played when the pose
        was measured. IDLE and PLANNER (--pre-pose-full): all of the yaw, against the frame of the pose's time, as the
        loco driver snaps its controller onto the robot before POSE. The robot does not play the stream then, and so
        POSE starts where the node's planner walked it. The rules that skip a seed are flags (SEED_FLAGS)."""
        a = self.args
        robot, gain = None, None
        p, outcome = self.seed_pose()
        if p is not None:
            xy = np.asarray(p['root_xy'][:2], float)
            jump = 0.0 if self.last_seed_xy is None else float(np.linalg.norm(xy - self.last_seed_xy))
            self.last_seed_xy = xy
            age = float(p.get('age_s', 0.0))
            if self.mode != MODE_POSE and a.pre_pose_full == 'on':
                robot, gain = ((xy[0], xy[1]), float(p['root_yaw']), age), 1.0
            elif self.mode == MODE_POSE and 0 < a.max_seed_jump < jump:
                outcome = 'jump'
                self.say(f'the SLAM pose jumped {jump:.2f} m in 0.2 s: no seed this time '
                         f"({self.skips['jump'] + 1} so far)")
            else:
                robot = ((xy[0], xy[1]), float(p['root_yaw']), age + a.track_delay)
        if outcome is not None:
            self.skips[outcome] += 1
        if self.trace is not None:
            self.trace_replan(F, p, robot)
        r = self.runner
        if self.release is not None and self.release['F'] is None:
            self.release['F'] = F
        seeds, kept_out, held_back = r.seeds, r.kept_out, r.held_back
        first, _ = r.replan(F, robot, yaw_gain=gain)
        if self.trace is not None and r.seeds > seeds:
            self.trace.write(json.dumps({'seed': r.last_seed, 'F': F}) + '\n')
        if self.trace is not None and r.kept_out > kept_out:
            self.trace.write(json.dumps({'kept_out': r.kept_out, 'F': F}) + '\n')
        self.seed_outcome = outcome or ('seeded' if r.seeds > seeds else 'hold' if r.held_back > held_back
                                        else 'keep_out' if r.kept_out > kept_out else None)
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
        if self.mode == MODE_POSE and self.args.strip_seed_turns == 'on':
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
        if self.release is not None and self.release['F'] is not None and self.cur > self.release['F']:
            self.release = None
            self.grasp.released = False
        if self.cur >= self.runner.next_replan:
            self.replan(self.cur)
        self.flush()
        self.update_fsm(self.frame_now())
        self.rearm_pick()
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
        if now - self.last['column'] >= COLUMN_S:
            self.last['column'] = now
            self.poll_column(now)
        if self.box_route is not None:
            self.box_route.step(now)
        if now - self.last['bus'] >= 0.1:
            self.last['bus'] = now
            self.publish_state()
        if now - self.last['ref'] >= 1.0 / REF_HZ:
            self.last['ref'] = now
            f = self.frame_now()
            self.bus.publish(REF_TOPIC, {'mode': self.mode, 'qpos': np.round(f.qpos[0:36], 4).tolist(),   # (36,)
                                         'held_box': None if f.ghost_box is None else np.round(f.ghost_box, 4).tolist(),
                                         'command': self.command_local().round(3).tolist()})              # (3, 4)

    def command_local(self):
        """The command as the controller reads it (the loco driver's gizmo): the simulation object 0.33, 0.67 and
        1.0 s ahead and its facing there, [dx, dy, ux, uy] each in the controller root's heading frame (x forward,
        y left). The controller runs ahead of the robot, so the window and UI 3 put it at the robot's map pose."""
        c = self.sim.controller
        yaw = qyaw(c.pose.rootRot)
        cy, sy = math.cos(yaw), math.sin(yaw)
        R = np.array([[cy, -sy], [sy, cy]])
        rel = (np.asarray(c.trajectory['pos'], float)[1:, :2] - np.asarray(c.pose.rootPos[:2], float)) @ R   # (3, 2)
        d = np.array([[math.cos(qyaw(q) - yaw), math.sin(qyaw(q) - yaw)] for q in c.trajectory['rot'][1:]])   # (3, 2)
        return np.concatenate([rel, d], axis=1)                                                              # (3, 4)

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
                self.release_hold(f"grip check: the reference lifted the box but it is still on the floor at "
                                  f"({meas['pos'][0]:.2f}, {meas['pos'][1]:.2f}) m", True, meas, by='grip')
            return
        self.grip.update(frame, w.holding_id, None)
        if w.regrasp == b['id'] and w.pick_id is None and meas['pos'][2] - self.args.box_size[2] / 2 < self.args.grip_phys_dz:
            b['pos'] = [float(meas['pos'][0]), float(meas['pos'][1])]
            b['yaw'] = box_yaw_near(meas['yaw'], b['yaw'], self.box_turn)
            if self.regrasp_wait == 'seen':
                self.regrasp_wait = 'loco'
                self.fsm_cause('the box is seen on the floor: it is picked up again from Locomotion')
                self.say(f"the box is seen on the floor at ({b['pos'][0]:.2f}, {b['pos'][1]:.2f}) m: the controller "
                         'picks it up again once the robot plays Locomotion')
            return
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
            self.say(f'the box moved {moved:.2f} m to ({pos[0]:.2f}, {pos[1]:.2f}) m: moved in the world too')

    # -- the grasp ----------------------------------------------------------------------------------------------
    def default_empty_gap(self):
        return EMPTY_GAP

    def empty_gap(self):
        return self.default_empty_gap() if self.args.empty_gap is None else float(self.args.empty_gap)

    def send_arm_tau(self, now):
        dt = CONTROL_DT if self.last['tau'] is None else min(0.1, now - self.last['tau'])
        self.last['tau'] = now
        frame = None if self.runner is None else self.frame_now()
        self.grasp.measure(self.robot.get(), frame)
        if frame is None or self.mode != MODE_POSE:
            self.grasp.ramp, self.grasp.on, self.grasp.cut = 0.0, False, False
            self.hands_empty_s = 0.0
            self.link.arm_tau(np.zeros(14))
            self.record_grasp(now, frame)
            return
        held = floor = None
        if frame.ghost_box is not None:
            held = to_heading_frame(frame.ghost_box[0:3], frame.qpos[0:2], pelvis_yaw(frame.qpos))
        meas = self.reports.fresh(self.args.reach_fresh_s)
        if meas is not None and self.slam is not None and meas['pos'][2] - self.args.box_size[2] / 2 < self.args.grip_phys_dz:
            floor = to_heading_frame(meas['pos'], self.slam['root_xy'], float(self.slam['root_yaw']))
        self.check_hands(dt, frame, now)
        self.link.arm_tau(self.grasp.step(dt, frame, held, floor))
        self.record_grasp(now, frame)

    def check_hands(self, dt, frame, now):
        """lmm-g1-box presses the palms only while the physical box is within reach of them, which its physics knows
        at every step. Here the camera's reports are late and miss a box at the feet or behind the arms, so the palms
        are measured instead. With the palm force on, palm sites closer together than --empty-gap hold no box between
        them: the force and the squeezes stop at once, before the hands meet, and stay off for the rest of this hold.
        The hold is dropped HANDS_EMPTY_S later. The palms open again without the force, so the drop waits for the
        time alone, not for the gap to stay small."""
        g, w = self.grasp, self.sim.world
        if frame.held is None or w.holding_id != frame.held or g.released:
            g.cut, self.hands_empty_s = False, 0.0
            return
        thr = self.empty_gap()
        if not g.cut and g.ramp > 0 and g.gap is not None and g.gap < thr:
            g.cut, g.ramp, self.hands_empty_s = True, 0.0, 0.0
            text = f'the palms are {g.gap:.3f} m apart, below {thr:.3f} m: no box between them, the palm force is off'
            lead = self.between_lead()
            if lead is not None:
                text += f' (the scan between the palms disagreed {lead:.2f} s before)'
            self.grasp_events.append((now, 'cut', text))
            self.say(text)
            return
        if not g.cut:
            return
        self.hands_empty_s += dt
        if self.hands_empty_s >= HANDS_EMPTY_S:
            self.hands_empty_s = 0.0
            self.release_hold(f'the palm gap fell below {thr:.3f} m {HANDS_EMPTY_S:g} s ago: no box between the hands',
                              True, by='palms')

    def record_grasp(self, now, frame):
        g = self.grasp
        held = frame is not None and frame.held is not None
        pz = None if g.palms is None else 0.5 * float(g.palms[0][2] + g.palms[1][2])
        self.grasp_trace.append((now, g.gap, g.ref_gap, self.empty_gap(), g.ramp if self.mode == MODE_POSE else 0.0,
                                 g.cut, held, pz, held and self.mode == MODE_POSE and not g.released))

    def poll_column(self, now):
        """The map server's newest LiDAR scan in a vertical column at the measured palms' midpoint: the top of what is
        between the palms, the box's top while the hands hold it. The maps cut every point near the robot, and the
        camera's reports miss a box in the hands or at the feet, the scan does not. Display only, nothing is dropped
        for it: a hold disagrees when the column tops more than COLUMN_MARGIN below the palms, or holds no point where
        the LiDAR sees it at the palms' height (a held box's top is not below the palms). The Mid-360 sees down to 52
        deg below its horizon only: standing, the column in front of the chest is seen from about 0.7 m up."""
        g = self.grasp
        if g.palms is None:
            return
        mid = 0.5 * (g.palms[0] + g.palms[1])
        r = self.map.request({'op': 'column', 'points': [[float(mid[0]), float(mid[1])]], 'r': COLUMN_R})
        if r is None:
            self.column_why = 'the map server does not answer'
            return
        if not r.get('localized'):
            self.column_why = r.get('why') or 'not localized'
            return
        self.column_why = None
        if r['stamp_ns'] == self.column_stamp:
            return
        self.column_stamp = r['stamp_ns']
        t = now - float(r['age_s'])
        row = next((x for x in reversed(self.grasp_trace) if x[0] <= t), None)
        pz, hold = (None, False) if row is None else (row[7], row[8])
        top, n = r['tops'][0], int(r['counts'][0])
        seen = (r.get('seen') or [None])[0]
        low = top is not None and pz is not None and top < pz - COLUMN_MARGIN
        empty = n == 0 and seen is not None and pz is not None and seen[0] <= pz <= seen[1]
        bad = bool(hold and (low or empty))
        self.column_trace.append((t, top, n, pz, hold, bad, None if seen is None else seen[0]))
        if bad:
            self.between_ok_since = None
            if self.between_since is None:
                self.between_since = t
                self.between_runs += 1
                what = (f'tops at {top:.2f} m, {pz - top:.2f} m below the palms ({pz:.2f} m)' if low else
                        f'holds no point, seen from {seen[0]:.2f} m up, the palms at {pz:.2f} m')
                self.grasp_events.append((t, 'between', f'the scan between the palms {what}: no box between them '
                                                        '(display only, nothing is dropped for it)'))
        elif self.between_since is not None:
            if not hold:
                self.between_since = self.between_ok_since = None
            elif top is not None:
                self.between_ok_since = t if self.between_ok_since is None else self.between_ok_since
                if t - self.between_ok_since >= COLUMN_AGREE_S:
                    self.between_since = self.between_ok_since = None

    def between_lead(self):
        """s since the scan between the palms began to disagree with this hold, or None."""
        return None if self.between_since is None else time.monotonic() - self.between_since

    def grasp_trace_state(self):
        """The last GRASP_TRACE_S of palm gaps for UI 1 and UI 3, every other control step, t in s before now."""
        now = time.monotonic()
        rows = list(self.grasp_trace)[::-2][::-1]
        r3 = lambda v: None if v is None else round(float(v), 3)
        cols = [c for c in self.column_trace if now - c[0] <= GRASP_TRACE_S]
        return {'t': [round(r[0] - now, 2) for r in rows], 'gap': [r3(r[1]) for r in rows],
                'ref': [r3(r[2]) for r in rows], 'thr': [r3(r[3]) for r in rows],
                'force': [round(float(r[4]), 2) for r in rows], 'cut': [bool(r[5]) for r in rows],
                'held': [bool(r[6]) for r in rows], 'pz': [r3(r[7]) for r in rows], 'hold': [bool(r[8]) for r in rows],
                'column': {'t': [round(c[0] - now, 2) for c in cols], 'top': [r3(c[1]) for c in cols],
                           'n': [c[2] for c in cols], 'pz': [r3(c[3]) for c in cols],
                           'hold': [bool(c[4]) for c in cols], 'bad': [bool(c[5]) for c in cols],
                           'seen_lo': [c[6] for c in cols],
                           'margin': COLUMN_MARGIN, 'r': COLUMN_R, 'why': self.column_why, 'runs': self.between_runs,
                           'since_s': None if self.between_lead() is None else round(self.between_lead(), 2)},
                'events': [[round(t - now, 2), k, text] for t, k, text in self.grasp_events if now - t <= GRASP_TRACE_S],
                'wall': round(time.time(), 3),
                'empty_gap': self.empty_gap(), 'default': self.default_empty_gap(), 'empty_s': HANDS_EMPTY_S,
                'width': min(self.args.box_size[:2]), 'span_s': GRASP_TRACE_S, 'last_drop': self.last_drop_state()}

    def last_drop_state(self):
        d = self.last_drop
        return None if d is None else {'by': d['by'], 'why': d['why'], 'ago_s': round(time.time() - d['wall'], 1),
                                       'between_s': d.get('between_s')}

    def release_hold(self, why, regrasp, meas=None, by='pose'):
        """The box left the hands: the world drops its hold and the palm force stops at once. A report of the box on
        the floor (meas, else a fresh one) puts it there, and with regrasp the controller picks it up again (the grip
        check of the page). Without one it is put on the floor under the palms, and with regrasp the pick waits for a
        report of it on the floor, else for X."""
        w, b = self.sim.world, self.world_box()
        if b is None or w.holding_id != b['id']:
            return False
        if meas is None:
            meas = self.reports.fresh(self.args.reach_fresh_s)
            if meas is not None and meas['pos'][2] - self.args.box_size[2] / 2 >= self.args.grip_phys_dz:
                meas = None
        if meas is not None:
            pos, yaw = [float(meas['pos'][0]), float(meas['pos'][1])], box_yaw_near(meas['yaw'], b['yaw'], self.box_turn)
        else:
            pos, yaw = self.palms_on_floor(), b['yaw']
        if not self.sim.drop(b['id'], pos, yaw):
            return False
        self.drops += 1
        lead = self.between_lead()
        self.last_drop = {'wall': time.time(), 'by': by, 'why': why, 'between_s': None if lead is None else round(lead, 2)}
        self.grasp_events.append((time.monotonic(), by, why))
        # A place target is for a held box (set_place needs one). The page's grip check comes during the rise, before
        # any place is asked for. Here the camera sees the box left on the floor some 2 s after the grasp, often after
        # the place was asked for, and the target kept made the controller put down the box it had just failed to lift.
        w.place_target = None
        # The frames made before the drop still hold the box, and with the reference mid-lift the controller grasps
        # again on the first frame it makes (the clip's held output is high), so the played hold never ends: the
        # force stays off until the frames of the next re-plan play.
        self.grasp.released = True
        self.grasp.ramp = 0.0
        self.grasp.cut, self.hands_empty_s = False, 0.0
        self.release = {'F': None, 'regrasp': bool(regrasp)}
        # The task goes through Locomotion before the pick again. The controller still in the pick's clip would grasp
        # again on its next frame (graspReach is horizontal only, and mid-lift the clip's held output is high), with
        # the hands up in the air: as a pick cancelled while bent down, the clip rises with empty hands (Stand up).
        w.pick_id = None
        c = self.sim.controller
        if regrasp and c.mode == 'box' and c.boxTag == 'pick':
            c.boxTag = 'place'
        self.regrasp_wait = None if not regrasp else 'loco' if meas is not None else 'seen'
        if not regrasp:
            w.regrasp = None
        where = f'({pos[0]:.2f}, {pos[1]:.2f}) m' + ('' if meas is not None else ', under the palms (not seen)')
        then = ('the robot stands up, and from Locomotion the controller picks it up again' if regrasp and meas is not None
                else 'the robot stands up, and the pick waits until the camera sees it on the floor' if regrasp
                else 'X picks it up again')
        self.fsm_cause(why, by)
        self.say(f'{why}: the hold is dropped, the box is at {where}, {then} ({self.drops} so far)')
        return True

    def rearm_pick(self):
        """After a drop: the pick again once the robot plays Locomotion (the frames made after the drop played) and
        the box's place is known from the camera."""
        w, c = self.sim.world, self.sim.controller
        if self.regrasp_wait != 'loco' or w.regrasp is None or w.pick_id is not None or w.holding_id is not None:
            return
        if self.release is not None or c.mode != 'loco' or self.fsm['state'] != 'locomotion':
            return
        b = w.find(w.regrasp)
        if b is None:
            self.regrasp_wait = None
            return
        w.pick_id = b['id']
        self.regrasp_wait = None
        self.sim.force_search = True
        self.fsm_cause('back in Locomotion after the drop: picking the box up again')
        self.say(f"back in Locomotion: the controller picks the box up again at ({b['pos'][0]:.2f}, {b['pos'][1]:.2f}) m")

    def palms_on_floor(self):
        """The floor under the palms' midpoint in the map frame (the controller's root without a SLAM pose)."""
        palms = self.grasp.palms
        if palms is None or self.slam is None:
            return list(self.sim.controller.pose.rootPos[:2])
        mid = 0.5 * (palms[0] + palms[1])
        xy, yaw = self.slam['root_xy'], float(self.slam['root_yaw'])
        c, s = math.cos(yaw), math.sin(yaw)
        return [xy[0] + c * mid[0] - s * mid[1], xy[1] + s * mid[0] + c * mid[1]]

    def leave_pose(self, why):
        """Out of POSE the node's planner (or damping) has the arms and the palm force is off, so a held box falls.
        The hold is dropped and the targets cleared: POSE starts again from locomotion, not from a pick, a place or a
        regrasp that the reference went on with while the robot stood."""
        self.release_hold(f'{why}: the arms let go of the box', False)
        if self.has_target() or self.sim.world.regrasp is not None:
            self.fsm_cause(f'{why}: the targets are cleared')
            self.cancel(from_task=True)

    # -- the box task's state for UI 3 ----------------------------------------------------------------------------
    def fsm_cause(self, why, by=None):
        self.fsm['why'] = (time.time(), why, by)

    def fsm_state(self, frame):
        """The task state at the frame the robot plays (FSM_STATES). 'lost': from a drop that picks the box up again
        until the frames made after it play, and while the box that left the hands is not seen on the floor."""
        w = self.sim.world
        if (self.release is not None and self.release['regrasp']) or (
                w.holding_id is None and w.regrasp is not None and self.regrasp_wait == 'seen'):
            return 'lost'
        sem = frame.tag['semantics']
        if sem in ('pick-pre', 'pick-post', 'place-pre', 'place-post'):
            return sem
        held = frame.held is not None
        if sem == 'navigate':
            return 'to_place' if held else 'to_box'
        return 'carry' if held else 'locomotion'

    def update_fsm(self, frame):
        f, now = self.fsm, time.time()
        s = self.fsm_state(frame)
        if s == f['state']:
            return
        cause = f['why'] if f['why'] is not None and now - f['why'][0] < 3.0 else None
        why = None if cause is None else cause[1]
        if why is None and f['state'] == 'lost':
            why = ('the frames made after the drop play: the pick clip rises with empty hands' if s == 'place-post'
                   else 'the frames made after the drop play')
        e, by = frame.tag.get('engaged'), None if cause is None else cause[2]
        if s in ('pick-pre', 'place-pre') and e is not None and e['id'] != f['engaged']:
            f['engaged'], by = e['id'], engage_by(e)
            words = engage_words(e, self.sim.controller.p['reapproachTime'])
            self.say(f'the box controller takes over the {"pick" if e["kind"] == "pick" else "put-down"} {words}')
            why = words + (f' (after {why})' if why else '')
        if f['state'] is not None:
            f['history'].append((round(now, 2), f['state'], s, why, by))
        f['state'], f['since'], f['why'] = s, now, None

    def fsm_signals(self, frame):
        w, g, b = self.sim.world, self.grasp, self.world_box()
        rest = self.args.box_size[2] / 2
        meas = self.reports.last
        age = None if meas is None else time.time() - meas['wall']
        xy = self.robot_xy()
        return {
            'mode': self.mode, 'holding': w.holding_id, 'pick': w.pick_id, 'regrasp': w.regrasp,
            'regrasp_wait': self.regrasp_wait,
            'place': None if w.place_target is None else [round(v, 2) for v in w.place_target['pos']],
            'ref_held': frame.held is not None,
            'ref_box_dz': None if frame.ghost_box is None else round(float(frame.ghost_box[2]) - rest, 3),
            'grip_ref_dz': GRIP_REF_DZ,
            'force_on': bool(g.on), 'ramp': round(g.ramp, 2), 'released': bool(g.released),
            'gap': None if g.gap is None else round(g.gap, 3), 'width': min(self.args.box_size[:2]),
            'ref_gap': None if g.ref_gap is None else round(g.ref_gap, 3),
            'gap_min': round(self.empty_gap(), 3), 'cut': bool(g.cut),
            'empty_s': round(self.hands_empty_s, 2), 'empty_need': HANDS_EMPTY_S,
            'last_drop': self.last_drop_state(),
            'grip_count': self.grip.count, 'grip_need': self.grip.need, 'grip_dz': self.args.grip_phys_dz,
            'report': None if meas is None else {
                'track': self.reports.track, 'age_s': round(age, 2), 'dz': round(float(meas['pos'][2]) - rest, 3),
                'dist': round(hypot(meas['pos'][0] - xy[0], meas['pos'][1] - xy[1]), 2)},
            'box_xy': None if b is None else [round(b['pos'][0], 2), round(b['pos'][1], 2)],
            'drops': self.drops}

    def match_state(self, frame):
        """The page's markers and its weights panel: the controller's last searches, their cost and the share of each
        feature group in it, and where the controller aims, in the map frame. `robot` is the frame the robot plays
        now, `plan` the newest frame sent (ahead_s ahead of it, made again at each re-plan), `controller` the
        controller's committed frame (at most one re-plan step ahead of the robot), `views` their clips' views."""
        c, w, cfg = self.sim.controller, self.sim.world, self.sim.cfg
        b = c.backend
        xyyaw = lambda p, yaw: [round(float(p[0]), 3), round(float(p[1]), 3), round(float(yaw), 3)]
        last = {}
        for kind, m in c.lastMatch.items():
            sizes = cfg['controllers'][QUERY_DB[kind]]['group_sizes']
            e = np.asarray(m['gain'], float) * (np.asarray(m['qn'], float) - np.asarray(m['features'], float)) ** 2   # (F,)
            err = np.add.reduceat(e, np.cumsum([0] + sizes[:-1]))                                             # (G,)
            total = float(err.sum())
            g = int(b.db[m['name']].idx[m['frame']])
            last[kind] = {'cost': round(float(m['cost']), 3), 'g': g, 'clip': self.clips.of(g), 'db': m['name'],
                          'share': [round(float(x) / total, 3) if total > 0 else 0.0 for x in err]}
        phase = c.box_phase()
        pre = phase in ('pick-pre', 'place-pre')
        t = c.target if c.target and (pre or c.mode == 'loco') else None
        a = c.anim_target() if pre else None
        r = c.commit if phase and not pre else None
        pb, pt = w.pick_box(), w.place_target
        g = frame.tag.get('g')
        cg = None if b.active is None else int(b.db[b.active].idx[b.frame])
        pg = self.runner.frame_at(self.runner.produced - 1).tag.get('g')
        robot, ctrl, plan = self.clips.of(g), self.clips.of(cg), self.clips.of(pg)
        p = c.p
        return {
            'robot': {'g': g, 'clip': robot, 'semantics': frame.tag['semantics'], 'holding': frame.held is not None},
            'controller': {'semantics': c.semantics, 'phase': phase, 'holding': w.holding_id is not None,
                           'in_use': 'loco' if c.mode == 'loco' else c.boxTag, 'g': cg, 'clip': ctrl, 'db': b.active},
            'plan': {'g': pg, 'clip': plan},
            'views': {str(x['i']): self.clips.view(x['i']) for x in (robot, ctrl, plan) if x is not None},
            'db_of': {k: c.db_of(k) for k in QUERY_DB},
            'radii': {'search': p['engageRadius'], 'engage': p['forceEngageRadius'], 'place_engage': p['placeEngageRadius'],
                      'reapproach': p['reapproachTime'], 'z_upright': p['zUpright'], 'upright_tilt': p['uprightTilt']},
            'engage': engage_view(c.engageCheck, p['reapproachTime']),
            'engaged': engage_view(c.engagedBy, p['reapproachTime']),
            'ahead_s': round(self.args.lookahead * CONTROL_DT, 2), 'last': last,
            'groups': {k: cfg['controllers'][QUERY_DB[k]]['groups'] for k in QUERY_DB},
            'weights': {k: [float(x) for x in c.p['weights'].get(k, [])] for k in QUERY_DB},
            'side_costs': [round(float(x), 3) for x in c.lastCosts],
            'enter': c.p['boxEnter'], 'leave': c.p['boxLeave'], 'side_margin': c.p['sideMargin'],
            'traj': [xyyaw(p, qyaw(q)) for p, q in zip(c.trajectory['pos'], c.trajectory['rot'])] if c.mode == 'loco' else None,
            'target': None if t is None else xyyaw(t['pos'], t['yaw']),
            'anim': None if a is None else xyyaw(a['pos'], math.atan2(a['dir'][1], a['dir'][0])),
            'rise': None if r is None else xyyaw(r['pos'], r['yaw']),
            'place': None if pt is None else xyyaw(pt['pos'], pt['yaw']),
            'walk_to': None if w.walk_to is None else [round(float(v), 3) for v in w.walk_to[:2]],
            'pick_box': None if pb is None else xyyaw(pb['pos'], pb['yaw']),
            'track': self.reports.track}

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
        match = None if cur is None else self.match_state(cur)
        if self.db_page is not None:
            self.db_page.live = {'ok': True, 'wall': round(time.time(), 2), 'mode': self.mode, 'loco': self.args.loco,
                                 'fsm': self.fsm['state'], 'phase': self.phase,
                                 'match': match}
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
            'seed_rules': None if r is None else {
                'outcome': self.seed_outcome, 'seeds': r.seeds,
                'skipped': {**self.skips, 'hold': r.held_back, 'keep_out': r.kept_out}},
            'phase': self.phase, 'task': task, 'match': match, 'grasp_trace': self.grasp_trace_state(),
            'db_port': None if self.db_page is None else self.db_page.port,
            'box_route': None if self.box_route is None else self.box_route.state(),
            'fsm': None if cur is None or self.fsm['state'] is None else {
                'state': self.fsm['state'], 'since_s': round(time.time() - self.fsm['since'], 1),
                'history': list(self.fsm['history']), 'signals': self.fsm_signals(cur)},
            'driver': 'box', 'ui': self.ui, 'frame': self.args.frame, 'keys_from': self.keys_from,
            'params': None if self.flags is None else self.flags.spec(),
            'variant_defaults': self.variant_defaults,
            'argv': None if self.flags is None else self.flags.argv(),
            'notices': [] if self.flags is None else list(self.flags.notices)})

    def close(self):
        if self.trace is not None:
            self.trace.close()
        try:
            self.link.arm_tau(np.zeros(14))
        except zmq.ZMQError:
            pass
        self.link.close()
        self.robot.close()
        if self.db_page is not None:
            self.db_page.close()
            self.db_page = None


# ---------------------------------------------------------------------- modes

def run_headless(args):
    br = Bridge(args)
    br.ui, br.box_route = 'headless', None
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
                if w.holding_id is None and (w.pick_id is not None or w.regrasp is not None):
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


def run_web(args, flags):
    br = Bridge(args)
    br.ui, br.flags = 'web', flags
    op = OP.Operator(br.map, 'box')
    br.say("--web: no window, UI 3 drives this bridge (the map server's /ui3 page, http://localhost:8780/ui3)")
    try:
        while True:
            now = time.monotonic()
            for c in op.poll(now):
                br.operator_command(c, br.close)
            br.drive_sticks(now, op)
            br.phase = br.sim.controller.semantics
            br.tick(now)
            time.sleep(0.003)
    except KeyboardInterrupt:
        pass
    finally:
        br.close()


def run_terminal(args, flags):
    br = Bridge(args)
    br.ui, br.flags = 'terminal', flags
    op = OP.Operator(br.map, 'box')
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
            now = time.monotonic()
            for c in op.poll(now):
                br.operator_command(c, br.close)
            if not br.walking:
                br.drive_sticks(now, op)
            br.phase = br.sim.controller.semantics
            br.tick(now)
            time.sleep(0.003)
    except KeyboardInterrupt:
        pass
    finally:
        br.close()


def page_value(name, v):
    """A page setting's flag value as the controller keeps it: 'off' is None, the menus' numbers are numbers."""
    if name in ('leave_moved', 'upright_tilt'):
        return None if v == 'off' else float(v)
    return v


def engage_by(e):
    return 'both' if e['by_cost'] and e['by_dist'] else 'cost' if e['by_cost'] else 'distance'


def engage_words(e, reapproach_time):
    """Why the box controller took over (the controller's engagedBy): its cost against Enter, its distance against
    the engage radius."""
    what = 'the box' if e['kind'] == 'pick' else 'the put-down marker'
    cost = f"cost {e['cost']:.2f} {'below' if e['by_cost'] else 'not below'} Enter {e['enter']:g}"
    dist = f"{e['dist']:.2f} m from {what}"
    if e['by_dist']:
        dist += f", within {e['radius']:g} m"
    elif e['dist'] < e['radius'] and e['reapproach'] is not None:
        dist += f", within {e['radius']:g} m but not by distance for {reapproach_time:g} s after going back to Locomotion"
    else:
        dist += f", not within {e['radius']:g} m"
    return f'by {"cost and distance" if engage_by(e) == "both" else engage_by(e)}: {cost}, {dist}'


def engage_view(e, reapproach_time):
    """UI 3's view of the controller's engageCheck (its last search near a target) or engagedBy (its last take-over)."""
    if e is None:
        return None
    v = {'kind': e['kind'], 'mode': e['mode'], 'enter': e['enter'], 'radius': e['radius'], 'search': e['search'],
         'cost': round(float(e['cost']), 3) if math.isfinite(e['cost']) else None, 'dist': round(float(e['dist']), 3),
         'wait_s': None if e['reapproach'] is None else round(max(0.0, reapproach_time - e['reapproach']), 2)}
    if 'by_cost' in e:
        v.update(id=e['id'], by=engage_by(e), text=engage_words(e, reapproach_time))
    return v


def page_choice(name, pv):
    if name in ('leave_moved', 'upright_tilt'):
        return 'off' if pv is None else f'{pv:g}'
    return pv


def weights(s):
    v = [float(x) for x in s.split(',')]
    if not all(math.isfinite(x) and x >= 0 for x in v):
        raise argparse.ArgumentTypeError('weights are numbers of 0 or more, comma separated')
    return v


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
    ap.add_argument('--wrist-kp', type=float, default=WRIST_YAW_KP,
                    help="the node's wrist yaw kp, which turns --wrist-squeeze into an arm_tau torque (20 with the lab's "
                         "--debug-arm-pd-gains)")
    ap.add_argument('--empty-gap', type=float, default=EMPTY_GAP,
                    help='m between the measured palm sites (the palm centres, 5.3 cm inside each hand): closer than '
                         'this while the palm force holds the box, the force and the squeezes stop at once, and the hold '
                         f'is dropped {HANDS_EMPTY_S:g} s later (default {EMPTY_GAP:g}: the box held is about 0.44 m, '
                         'the hands touch at about 0.11 m). Set live from the Palm gap card of UI 1 and UI 3')
    g = ap.add_argument_group("lmm-g1-box's settings", "the page's Play panel (UI 3's Settings tab), each with its default")
    g.add_argument('--palm-target', choices=('box', 'mid'), default='box',
                   help='the palm force pulls toward the box centre or toward each other (the midpoint of the palms)')
    g.add_argument('--shoulder-squeeze', type=float, default=0.0,
                   help='rad the shoulder rolls turn inward while the reference holds the box')
    g.add_argument('--wrist-squeeze', type=float, default=0.0,
                   help='rad the wrist yaws turn inward while the reference holds the box')
    g.add_argument('--hold', choices=tuple(HOLDS), default='low',
                   help='pause the anchoring: low (while the reference is low), box (while the box controller runs), '
                        'step (from the last step of the box clip), none (never)')
    g.add_argument('--foot-lock', choices=('on', 'off'), default='on', help='foot locking (IK) of the reference')
    g.add_argument('--pick-speed', type=float, default=1.0,
                   help='the box clips (walk in, bend, grasp or release, rise) play at this share of their speed')
    g.add_argument('--pick-slow', choices=('clip', 'step'), default='clip',
                   help='slowed down: all of the box clips, or from the last step to the first step after')
    g.add_argument('--warp', choices=('always', 'step', 'pickstep'), default=None,
                   help="root warping: through the walk in and the bend, only before the last step, or only before "
                        "the last step of a pick (default: the variant's)")
    g.add_argument('--leave-moved', choices=('off', '0.25', '0.5', '0.75', '1'), default=None,
                   help="m: back to Locomotion when the target moves this far before the bend (default: the variant's)")
    g.add_argument('--box-entry', choices=('walk', 'any'), default='walk',
                   help='box search: not into the end of a walk or the middle of a SceneBot clip, or every frame')
    g.add_argument('--rise', choices=('finish', 'search'), default=None,
                   help="rising after a grasp or a release: with the same clip, or searching every 0.1 s "
                        "(default: the variant's)")
    g.add_argument('--transition-cost', type=float, default=None,
                   help="added to the cost of every frame but the playing one (default: the variant's)")
    g.add_argument('--upright-tilt', choices=('off', '45', '30', '20'), default=None,
                   help="deg: back to Locomotion after a pick or a place once the pelvis is above 0.72 m and the torso "
                        "is within this of vertical (default: the variant's)")
    g.add_argument('--box-db', choices=('same', 'different'), default='same',
                   help='one Box Matching Database for pick up and put down (with the tag), or one for each')
    for kind, what in (('loco', 'Locomotion'), ('pick', 'pick up'), ('place', 'put down')):
        g.add_argument(f'--weights-{kind}', type=weights, default=None,
                       help=f"the {what} query's feature group weights, comma separated (default: the variant's)")
    g.add_argument('--box-enter', type=float, default=None, help='the box controller takes over below this cost')
    g.add_argument('--box-leave', type=float, default=None, help='back to Locomotion above this cost')
    g.add_argument('--side-margin', type=float, default=None,
                   help='another side must cost less than the current side minus this')
    g.add_argument('--engage-radius', type=float, default=None,
                   help='m: the box database is searched for each side of the box or the put-down marker within this '
                        "of it (the page's 3)")
    g.add_argument('--force-engage-radius', type=float, default=None,
                   help="m: the box controller takes over a pick within this of the box, whatever the cost, and leaves "
                        "only beyond this + 0.3 m (the page's 1.1, 0: by the cost only)")
    g.add_argument('--place-engage-radius', type=float, default=None,
                   help="m: the same for a put-down, within this of the marker (default: the variant's)")
    g.add_argument('--reapproach-time', type=float, default=None,
                   help="s: after the box controller goes back to Locomotion before the bend, the distance does not take "
                        "over again for this long (the page's 1)")
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
    g.add_argument('--yaw-gain', type=float, default=1.0,
                   help="POSE: share of the yaw gap between the SLAM pelvis and the reference a seed turns the "
                        "controller (the page's snap-xyyaw: 1, snap-xy: 0). The robot does not turn with a seed (see send)")
    ap.add_argument('--track-delay', type=float, default=0.1,
                    help='s the robot plays behind the frame sent, added to the pose age when the seed looks back for the '
                         'frame to compare (SONIC\'s tracking delay: the leg joints match the frames 4-6 back in the sim). '
                         '0: none')
    ap.add_argument('--max-seed-jump', type=float, default=0.5,
                    help='m: in POSE a SLAM pose this far from the one 0.2 s before is not seeded from. 0: no check')
    ap.add_argument('--skip-stopped', choices=('on', 'off'), default='on',
                    help='on: no seed while the robot is stopped (O, damping). off: seeds then too, as before POSE')
    ap.add_argument('--skip-stale', choices=('on', 'off'), default='on',
                    help='on: no seed without a SLAM pose from the last 0.5 s. off: the newest pose, however old')
    ap.add_argument('--keep-out', choices=('on', 'off'), default='on',
                    help="on: in Locomotion no seed puts the controller's simulation object inside a box's collision "
                         "zone (half the box + the controller's collisionRadius, 0.36 m). The controller pushes it out "
                         "again within 0.1 s, and next to a placed box such seeds made the reference shuffle. Not the page's")
    ap.add_argument('--pre-pose-full', choices=('on', 'off'), default='on',
                    help="on: before POSE (IDLE, PLANNER) a seed turns all of the yaw gap, against the frame of the "
                         "pose's time. off: as in POSE, --yaw-gain of it, against that frame + --track-delay")
    ap.add_argument('--strip-seed-turns', choices=('on', 'off'), default='on',
                    help="on: in POSE the heading sent leaves out the seeds' turns since POSE started, so the robot does "
                         "not turn with a seed. off: sent with them, the robot turns with each seed (before 2026-11-05, "
                         "with a yaw gap at the start of POSE the robot and the reference turned on and on). Switched in "
                         "POSE, the robot turns by the seeds' turns so far")
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
    ap.add_argument('--db-port', type=int, default=8790,
                    help="HTTP port of the database page (sonic_g1_box/db_page.py), 0: none. Not in headless runs")
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
    ap.add_argument('--web', action='store_true',
                    help="no window: UI 3 (the map server's /ui3 page) sends the keys, the box keys and the mode commands")
    ap.add_argument('--smoke-frames', type=int, default=None, help='(testing) close the window after N drawn frames')
    args = ap.parse_args()
    flags = OP.Flags(ap, args, LIVE_FLAGS, RESTART_FLAGS)
    if args.headless is not None:
        run_headless(args)
    elif args.web:
        run_web(args, flags)
    elif args.terminal:
        run_terminal(args, flags)
    else:
        from box_window import run_window
        br = Bridge(args)
        br.ui, br.flags = 'window', flags
        run_window(br, args)


if __name__ == '__main__':
    main()
