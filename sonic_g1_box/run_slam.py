"""The box task (pick, carry, place, with or without the walk over) on the real
G1 with SLAM localization from motionmatching-g1-loco's live stack.

The scenario: the box stands about 2 m in front of the start pose. SLAM is
initialized and the map server anchors its frame at the start pose (that frame
is this matcher's world: origin at the start pose, facing +x). Nothing is
mapped. Once, after the anchor, the box pose is fixed from what the map server
reports (`objects`: Boxer's boxes in the map frame, or the sim's true box with
`live.server --objects truth`). From then on the box relative to the robot
follows from that fixed pose and the SLAM pose, so the lift needs no camera.
`m` walks over to that fixed box and picks it; `b` plays the pick, carry and
put-down where the robot stands and needs no box pose (nor the fix, nor
Boxer). Every --replan-s the stream is cut at the frame the robot plays and the
matcher's root is put on the SLAM pelvis (not during the ridden pick and place:
those play open loop from the committed pose). After m's pick the grip is
judged once from what the server still sees on the floor at the pick spot:
lost means stand still, no second try (b's pick is not judged).

    python sonic_g1_box/run_slam.py                       # window
    python sonic_g1_box/run_slam.py --terminal            # terminal keys, no window
    python sonic_g1_box/run_slam.py --headless 240 --auto-start --start-after-anchor

The window (live.mm_driver's UI 2, with the box): the measured robot at its SLAM
pelvis pose, the stick figure the reference the matcher streams (gray idle, blue
planner, orange POSE), the box where the reference has it (faint until x fixes
it), the objects the map server reports (amber: the one x would take), and with T
the command trajectory and m's route and stance.

Window keys:
  ]  start (planner mode, the robot stands)    P  planner <-> POSE
  X  fix the box from the map server           B  pick right here (no box pose needed), or place
  M  walk over to the fixed box + pick (M again cancels the walk)
  W A S D  move, in POSE only (heading frame; F: camera frame)   arrows  face   Shift  run
  T  gizmos    Enter  re-send the mode command    O  stop (damping, final)    Esc  quit

Terminal keys (type the letter, then Enter):
  ]  start (planner mode, the robot stands)    p  planner <-> POSE
  x  fix the box from the map server           b  pick right here (no box pose needed), or place
  m  walk over to the fixed box + pick (m again cancels the walk)
  w  walk forward (--walk-speed) / stop        o  stop (damping, final)       q  quit

Headless: waits for the node and the anchor, starts, settles, fixes the box
(not with --pick-here), enters POSE, triggers the pick (m, or b with
--pick-here), judges the grip (m's pick), carries --carry-s seconds forward,
places, stands.
"""

import argparse
import math
import os
import select
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
LOCO = os.environ.get('MM_LOCO', os.path.join(os.path.dirname(ROOT), 'motionmatching-g1-loco'))
sys.path.insert(0, LOCO)

import msgpack
import numpy as np
import zmq

from ego_slam import wire
from mm_g1 import config as C
from mm_g1 import quat
from mm_g1.features import yaw_quat
from mm_g1.states import State
from mm_stream import MMMotion, POLICY_FPS, fill_speeds, fit_rows
import ref_modes as RM
from run_hardware import (DeployLink, RobotState, build_matcher, MODE_IDLE, MODE_PLANNER, MODE_POSE,
                          MODE_STOPPED)

STATE_TOPIC = '/mm_driver/state'        # what UI 1 and the demo recorder read
GRID_DIM = 17 * 11
# MuJoCo joint order within qpos[7:36] (run_grasp.py): the joints the squeeze biases
L_SHOULDER_ROLL, R_SHOULDER_ROLL, L_WRIST_YAW, R_WRIST_YAW = 16, 23, 21, 28
L_SHOULDER_PITCH, R_SHOULDER_PITCH = 15, 22
L_ELBOW, R_ELBOW = 18, 25
ARMS = slice(7 + 15, 7 + 29)                 # both arms in qpos (MuJoCo order)
UPPER = slice(7 + 12, 7 + 29)                # the waist (3) and both arms: held through the carry


class SqueezedMotion(MMMotion):
    """The stream with the grasp bias of the box repo's run_grasp.py applied to the
    reference joints: while either hand's contact label is on, or the matcher holds
    the box (the carry frames carry no hand labels), the shoulder rolls press inward
    and the wrist yaws toe in, so tracking the reference clamps the box instead of
    only touching it; during a pick or place ride with both labels off and no box
    held the shoulders swing out to clear the box."""

    ARM_LEVER = 0.30                       # m of lateral wrist shift per rad of shoulder roll (measured on the model)

    def __init__(self, matcher, commander, shoulder_squeeze, wrist_squeeze, shoulder_open, shoulder_pitch=0.0,
                 elbow=0.0, hold_arms=True, center_hands=True):
        self.bias = (shoulder_squeeze, wrist_squeeze, shoulder_open, shoulder_pitch, elbow)
        self.hold_arms = hold_arms
        self.held_arms = None                  # the pick clip's final upper body, kept through the carry
        self.center_hands = center_hands
        self.center_roll = 0.0                 # both shoulder rolls, from the box's lateral offset at the pick entry
        self._last_state = None
        self._sbuf = self._sjbuf = self._svbuf = None
        self._s_built = 0                      # frames whose biased rows are up to date
        self._held_after = []                  # held_arms after each frame: where a later pass resumes
        self._redo_from = None
        super().__init__(matcher, commander)

    REBIAS_TAIL = 1000                     # frames a centre-roll change re-biases: all that can still be sent or drawn

    def _step_matcher(self):
        m = self.matcher
        before, roll = m.state, self.center_roll
        super()._step_matcher()
        if self.center_hands and m.state is State.PICK and before is not State.PICK and m.pick_at_stance:
            # the robot stops beside the stance, not on it: the box's lateral offset from the
            # pelvis at this moment (the SLAM pose against the fixed box) becomes a roll on
            # both shoulders, so the clamp closes where the box is (m's pick only: b's pick
            # does not use the box pose)
            left = float(quat.inv_mul_vec(m.rootRot, m.boxPos - m.rootPos)[1])
            self.center_roll = float(np.clip(left / self.ARM_LEVER, -0.5, 0.5))
            print(f'[slam-box] pick entry: the box is {left:+.2f} m left of the pelvis, '
                  f'shoulder rolls {self.center_roll:+.2f} rad to centre the hands', flush=True)
        elif m.state in (State.LOCOMOTION, State.MOVE_TO_PICK):
            self.center_roll = 0.0
        if self.center_roll != roll:           # the roll applies to every ride frame, not only the new ones
            self._redo_from = max(0, len(self._frames) - self.REBIAS_TAIL)

    def _rebuild(self):
        """The biases, from the first frame that changed on (the base class's appends and
        truncates, or the tail a centre-roll change asks for). The hold state is
        sequential, so a pass resumes from the state the previous one left after its
        last kept frame."""
        super()._rebuild()
        n = len(self._frames)
        if not (any(self.bias) or self.hold_arms):
            self.sent_qpos = self._qpos            # the frames the stream carries (the window draws them)
            return
        lo = min(self._s_built, self._changed_from, n)
        if self._redo_from is not None:
            lo, self._redo_from = min(lo, self._redo_from), None
        width = self._qbuf.shape[1]
        self._sbuf = q = fit_rows(self._sbuf, n, width)
        self._sjbuf = fit_rows(self._sjbuf, n, self._jbuf.shape[1])
        self._svbuf = fit_rows(self._svbuf, n, self._jbuf.shape[1])
        del self._held_after[lo:]
        q[lo:n] = self._qbuf[lo:n]
        sq, wq, op, sp, eb = self.bias
        cr = self.center_roll
        held_arms = self._held_after[lo - 1] if lo > 0 else None
        prev_state = self._meta[lo - 1][2] if lo > 0 else None
        for i in range(lo, n):
            _, _, state, held, (lc, rc) = self._meta[i]
            if cr and (state in (State.PICK, State.PLACE) or (state is State.CARRY and held)):
                q[i, 7 + L_SHOULDER_ROLL] += cr      # both arms toward the box (+: left)
                q[i, 7 + R_SHOULDER_ROLL] += cr
            if self.hold_arms:
                # the single SceneBot motion: after its pick the carry search would swap
                # in other clips' arms and the box drops, so the arms stay where the pick
                # left them until the reverse clip (the place) takes over
                if state is State.CARRY and prev_state is State.PICK and held_arms is None:
                    # the waist and both arms of the pick's last frame: the carry frames' torso
                    # moved the hands off the box once the arms alone were held (the pelvis
                    # orientation stays the carry frames': frozen, the robot walked off)
                    held_arms = q[i - 1, UPPER].copy()
                if state is State.CARRY and held_arms is not None:
                    q[i, UPPER] = held_arms
                elif state is not State.CARRY:
                    held_arms = None if state is not State.PICK else held_arms
                prev_state = state
            if state in (State.PICK, State.PLACE):
                # the pelvis ends some 20 cm behind the clip's during the squat and the pose
                # stream carries no root position, so only the arms can close the gap: a
                # positive pitch bias swings the reaching arms down, a negative elbow bias
                # straightens them (forward and down along the forearm)
                q[i, 7 + L_SHOULDER_PITCH] += sp
                q[i, 7 + R_SHOULDER_PITCH] += sp
                q[i, 7 + L_ELBOW] += eb
                q[i, 7 + R_ELBOW] += eb
            if lc or rc or held:                 # the carry frames carry no hand labels: keep clamping
                q[i, 7 + L_SHOULDER_ROLL] -= sq
                q[i, 7 + R_SHOULDER_ROLL] += sq
                q[i, 7 + L_WRIST_YAW] -= wq
                q[i, 7 + R_WRIST_YAW] += wq
            elif state in (State.PICK, State.PLACE) and not held:
                q[i, 7 + L_SHOULDER_ROLL] += op
                q[i, 7 + R_SHOULDER_ROLL] -= op
            self._held_after.append(held_arms)
        self.held_arms = held_arms
        from sonic_tracking import params as P
        self._sjbuf[lo:n] = q[lo:n, 7:36][:, P.MUJOCO_TO_ISAACLAB]
        fill_speeds(self._svbuf, self._sjbuf, lo, n)
        self._s_built = n
        self.sent_qpos = q[:n]                     # the frames the stream carries (the window draws them)
        self._joint_pos, self._joint_vel = self._sjbuf[:n], self._svbuf[:n]


class PalmForce:
    """--palm-force: while a hand's contact label is on at the frame the robot plays (the
    pick from its touch, the whole hold, the put-down until its release), that palm is
    pressed toward the midpoint between the palms with F newtons through its arm's
    Jacobian, tau = J^T f, at the robot's MEASURED joints (g1_debug). The torques go to
    the node on the arm_tau topic, which adds tau / kp to SONIC's arm targets with its
    own final gains. The midpoint needs no box pose: the palms press toward each other,
    on the box's sides. The palm is the sim's pad point on each wrist yaw link
    (run_grasp.py --palm-target mid, the same force in the box repo's sim)."""

    PALM = np.array([0.10, 0.007, 0.0])        # on the wrist yaw link; y mirrored for the right hand
    RAMP_S = 0.15                              # s for a hand's force to come in or go out
    TAU_MAX = 20.0                             # N*m per joint

    def __init__(self, force):
        import mujoco
        self.mj = mujoco
        self.force = float(force)
        m = self.model = mujoco.MjModel.from_xml_path(C.SCENE_BOX_SCENEBOT_XML if C.SCENEBOT_PICK else C.SCENE_BOX_XML)
        self.data = mujoco.MjData(m)
        self.wrist = [m.body(f'{side}_wrist_yaw_link').id for side in ('left', 'right')]
        # the arm joints 15..28 of qpos[7:36] (MuJoCo order) and their dofs
        arm = sorted((m.jnt_qposadr[j], m.jnt_dofadr[j]) for j in range(m.njnt)
                     if ARMS.start <= m.jnt_qposadr[j] < ARMS.stop)
        self.dof = [d for _, d in arm]
        self.jacp = np.zeros((3, m.nv))

    def torques(self, body_q, ramp):
        """(14,) arm torques at the measured joints body_q (29, MuJoCo order); ramp (2,)
        scales each hand's force (0..1). The base pose does not matter: J and f turn together."""
        mj, m, d = self.mj, self.model, self.data
        d.qpos[:] = m.qpos0
        d.qpos[7:36] = body_q
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        palms = [d.xpos[b] + d.xmat[b].reshape(3, 3) @ (self.PALM * [1.0, sign, 1.0])
                 for b, sign in zip(self.wrist, (1.0, -1.0))]
        mid = 0.5 * (palms[0] + palms[1])
        tau = np.zeros(14)
        for h in range(2):
            n = mid - palms[h]
            L = float(np.linalg.norm(n))
            if ramp[h] <= 0.0 or L < 1e-6:
                continue
            mj.mj_jac(m, d, self.jacp, None, palms[h], self.wrist[h])
            arm = slice(7 * h, 7 * h + 7)
            tau[arm] = self.jacp[:, self.dof[arm]].T @ ((ramp[h] * self.force / L) * n)
        return np.clip(tau, -self.TAU_MAX, self.TAU_MAX)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_of(q):
    return float(np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))


def rot2(v, a):
    c, s = np.cos(a), np.sin(a)
    out = np.array(v, float)
    out[0], out[1] = c * v[0] - s * v[1], s * v[0] + c * v[1]
    return out


class MapLink:
    """REQ client of live.server (REP 5591); every call returns None instead of blocking."""

    def __init__(self, endpoint, timeout_ms=60, retry_s=0.0):
        self.endpoint, self.timeout_ms = endpoint, timeout_ms
        self.retry_s = retry_s                 # after a request that got no answer, ask again only this much later
        self._down_until = 0.0
        self._open()

    def _open(self):
        self.sock = zmq.Context.instance().socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.connect(self.endpoint)

    def request(self, q):
        if self.retry_s and time.monotonic() < self._down_until:
            return None
        try:
            self.sock.send(msgpack.packb(q, use_bin_type=True))
            r = msgpack.unpackb(self.sock.recv(), raw=False)
            return r if r.get('ok') else None
        except zmq.ZMQError:
            self.sock.close(0)
            self._open()
            self._down_until = time.monotonic() + self.retry_s
            return None


def seed_from_slam(m, xy, yaw):
    """The matcher's root onto the robot's pelvis (map frame). World vectors turn
    with the heading; a held box rides the root, a resting box stays where it is."""
    m.rootPos[0:2] = xy
    dyaw = wrap(yaw - m.rootYaw)
    m.rootYaw = wrap(m.rootYaw + dyaw)
    m.rootRot = yaw_quat(m.rootYaw)
    for name in ('desiredDir', 'rootVel', 'rootAcc', 'cmdVel', 'cmdFace'):
        setattr(m, name, rot2(getattr(m, name), dyaw))


def pick_object(objects, expect_xy, radius):
    """The reported object nearest to where the box is expected, within radius."""
    best = None
    for o in objects or []:
        c = np.asarray(o['bbox_center_world'], float)
        d = float(np.hypot(c[0] - expect_xy[0], c[1] - expect_xy[1]))
        if d <= radius and (best is None or d < best[0]):
            best = (d, o)
    return None if best is None else best[1]


class SlamBoxBridge:
    def __init__(self, args):
        self.args = args
        if args.approach_scale != 1.0:
            # the real robot lags the reference's decelerations and overshoots a fast stop
            # into the box: slower approach speeds, and the servo stops commanding earlier
            k = float(args.approach_scale)
            C.MOVE_ROUTE_SPEED_MAX *= k
            # the endgame floor stays at the data's 0.35 m/s: below it the matcher picks
            # near-standing frames (the slow-walk dead zone) and the robot stops short
            C.MOVE_END_SPEED_MAX = max(C.MOVE_END_SPEED_MAX * k, C.MOVE_END_SPEED_MIN)
            C.MOVE_FWD_TOL = args.fwd_tol
            # the route aims MOVE_OVERSHOOT past the stance to keep the walk alive: with the
            # SceneBot box that is the box itself, and the robot runs into it. The slow walk
            # needs no overshoot, and the servo stops commanding earlier so the robot's lag
            # lands it on the stance
            C.MOVE_OVERSHOOT = 0.10
            C.MOVE_STOP_DIST = 0.30
            print(f'[slam-box] approach speeds x{k:.2f} (route up to {C.MOVE_ROUTE_SPEED_MAX:.2f} m/s, endgame '
                  f'{C.MOVE_END_SPEED_MIN:.2f} to {C.MOVE_END_SPEED_MAX:.2f} m/s, overshoot {C.MOVE_OVERSHOOT:.2f} m, '
                  f'commands stop {C.MOVE_STOP_DIST:.2f} m before the stance'
                  + (f', the pick waits for the root within {C.MOVE_FWD_TOL:.2f} m along the rail' if C.MOVE_FWD_TOL else '')
                  + ')', flush=True)
        if args.arrive_near is not None:
            # a physical grasp needs the robot where the clip's stance is: the pick fires
            # only inside this radius (and the loose fallback at twice it, after 2 s)
            C.MOVE_ARRIVE_NEAR = float(args.arrive_near)
            C.MOVE_ARRIVE_LOOSE = 2.0 * float(args.arrive_near)
            C.MOVE_ARRIVE_LOOSE_S = 2.0
            C.MOVE_ARRIVE_SPEED = min(C.MOVE_ARRIVE_SPEED, 0.12)
            print(f'[slam-box] the pick fires within {C.MOVE_ARRIVE_NEAR:.2f} m of the stance '
                  f'(settled under {C.MOVE_ARRIVE_SPEED:.2f} m/s), or within {C.MOVE_ARRIVE_LOOSE:.2f} m after 2 s', flush=True)
        self.matcher = build_matcher(args)            # world = the map frame: start pose at the origin, +x
        if args.stance_bias:
            # the robot drifts back from the reference while it squats; a stance this much
            # closer puts the hands round the box's middle instead of its near top edge
            self.matcher.stance_box_off[0] += args.stance_bias
            print(f'[slam-box] pick stance {-args.stance_bias:+.2f} m closer to the box '
                  f'(box {self.matcher.stance_box_off[0]:.2f} m ahead of the stance)', flush=True)
        self.cmd_vel, self.cmd_face = np.zeros(3), np.zeros(3)
        self.motion = SqueezedMotion(self.matcher, lambda m: (self.cmd_vel.copy(), self.cmd_face.copy()),
                                     args.shoulder_squeeze, args.wrist_squeeze, args.shoulder_open,
                                     args.shoulder_pitch_bias, args.elbow_bias, not args.no_hold_arms,
                                     not args.no_center_hands)
        self.link = DeployLink(args.bind, args.port)
        self.robot = RobotState(args.robot_host, args.robot_port)
        self.map = MapLink(args.map_rep)
        self.bus = wire.BusPublisher(args.bus_pub)
        self.mode = MODE_IDLE
        self.t0 = None
        self.cur_frame = 0
        self.published = -1
        self.pending_from = None
        self.next_replan = None
        self.slam = None
        self.sync_err = self.heading_err = None
        self.snaps_skipped = 0
        self.last_slam_xy = None
        self.box = {'fixed': False, 'source': None, 'size': None, 'detected_yaw': None}
        self.lost = False
        self.grip = None                                # (t_start, sightings) while the grip is judged
        self.grip_result = None                         # 'held', 'lost', or 'not checked' (b's pick)
        self.phase = 'idle'
        self.last = {'heartbeat': -1.0, 'planner': -1.0, 'bus': -1.0, 'arm_tau': None}
        self.walking = False
        self.palm = PalmForce(args.palm_force) if args.palm_force > 0 else None
        self.palm_ramp = [0.0, 0.0]
        self.palm_tau = np.zeros(14)

    # -- the map server ----------------------------------------------------------
    def slam_pose(self):
        r = self.map.request({'op': 'pose'})
        p = None if r is None else r.get('pose')
        self.slam = p if (p is not None and p.get('age_s', 9.0) < 0.5) else None
        return self.slam

    def status(self):
        return self.map.request({'op': 'status'})

    def objects(self):
        r = self.map.request({'op': 'objects'})
        return None if r is None else r.get('objects', [])

    def fix_box(self):
        """The box pose from the map server, once: the object nearest to the belief
        (--box-fwd / --box-lat from the start pose) within --box-radius."""
        m = self.matcher
        expect = m.rootPos[0:2] * 0 + np.array([C.BOX_SPAWN_FWD, C.BOX_SPAWN_LAT])
        o = pick_object(self.objects(), expect, self.args.box_radius)
        if o is None:
            return False
        c = np.asarray(o['bbox_center_world'], float)
        if self.args.fix_range_bias:
            # the detector puts the box's centre toward the camera (the sim: 4 to 5 cm at 2 m):
            # push the fix away from where the robot stands by that much
            away = c[0:2] - m.rootPos[0:2]
            c = c.copy()
            c[0:2] += self.args.fix_range_bias * away / (np.linalg.norm(away) + 1e-9)
        T = np.asarray(o['T_world_object'], float).reshape(4, 4)
        yaw = float(np.arctan2(T[1, 0], T[0, 0]))
        size = [float(v) for v in o['bbox_size_xyz']]
        if size[1] > size[0] * 1.15:
            # the detector's box has its long side along its own y: the library's box is long
            # along x (the SceneBot box, 0.3 x 0.2), so turn it a quarter and swap the extents
            yaw += 0.5 * np.pi
            size = [size[1], size[0], size[2]]
        # the box's symmetry: the fold nearest the belief's orientation (2 folds for the
        # SceneBot box, 4 for a cube), so the detected yaw does not send the approach round the side
        folds = int(getattr(C, 'SCENEBOT_ROT_FOLDS', 2)) if getattr(C, 'SCENEBOT_PICK', False) \
            else int(getattr(C, 'BOX_ROT_FOLDS', 1))
        period = 2.0 * np.pi / max(folds, 1)
        yaw = (yaw + 0.5 * period) % period - 0.5 * period
        m.boxPos = np.array([c[0], c[1], C.BOX_REST_Z])
        m.boxRot = quat.mul(yaw_quat(yaw - yaw_of(m.box_spawn_rot)), m.box_spawn_rot)
        self.box.update(fixed=True, source=str(o.get('language_label') or 'object'), size=size,
                        xy=[float(c[0]), float(c[1])], detected_yaw=yaw, track_id=o.get('track_id'),
                        bottom=float(c[2] - 0.5 * size[2]))
        lib_size = [2.0 * v for v in C.BOX_HALF]
        note = '' if max(abs(a - b) for a, b in zip(sorted(size), sorted(lib_size))) < 0.1 else \
            f' (the motion library is baked for a {lib_size[0]:.2f} x {lib_size[1]:.2f} x {lib_size[2]:.2f} m box)'
        print(f'[slam-box] box fixed at ({c[0]:.2f}, {c[1]:.2f}) m, yaw {math.degrees(yaw):.0f} deg, '
              f'size {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m, from {self.box["source"]}{note}', flush=True)
        return True

    def request_fix(self):
        """x: fix the box, or say why not."""
        if not self.fix_box():
            print('[slam-box] no object near the belief: is Boxer running, is the map anchored?', flush=True)

    # -- modes -------------------------------------------------------------------
    def start(self):
        if self.mode == MODE_STOPPED:
            print('[slam-box] stopped (damping); restart deploy.sh to resume', flush=True)
            return
        self.link.command(start=True, planner=True)
        self.mode = MODE_PLANNER
        print('[slam-box] start=1 planner=1 -> PLANNER (stand)', flush=True)

    def toggle_pose(self):
        if self.mode == MODE_PLANNER:
            self.link.command(planner=False)
            self.mode = MODE_POSE
            self.next_replan = None
            print('[slam-box] planner=0 -> POSE (the matcher drives the robot)', flush=True)
        elif self.mode == MODE_POSE:
            self.link.command(planner=True)
            self.mode = MODE_PLANNER
            print('[slam-box] planner=1 -> PLANNER (stand)', flush=True)
        else:
            print('[slam-box] press ] first', flush=True)

    def stop(self):
        self.link.command(stop=True, planner=self.mode == MODE_PLANNER)
        self.mode = MODE_STOPPED
        print('[slam-box] stop=1 -> damping', flush=True)

    def box_action(self):
        """b: the pick right where the robot stands, or the place while carrying."""
        if self.lost:
            print('[slam-box] the box was lost: standing, no second try', flush=True)
            return
        m = self.matcher
        self.matcher.trigger_box()
        print('[slam-box] ' + ('place' if m.state is State.CARRY else 'pick right here (the box pose is not used)')
              + f' requested (state {m.state.name})', flush=True)

    def move_pick(self):
        """m: walk over to the fixed box and pick it; m during the walk cancels it."""
        if self.lost:
            print('[slam-box] the box was lost: standing, no second try', flush=True)
            return
        m = self.matcher
        if m.state not in (State.LOCOMOTION, State.MOVE_TO_PICK):
            print(f'[slam-box] m only from locomotion (state {m.state.name}); b places while carrying', flush=True)
            return
        if m.state is State.LOCOMOTION and not self.box['fixed']:
            print('[slam-box] the box is not fixed (x): walking over to the belief '
                  f'{C.BOX_SPAWN_FWD:.2f} m ahead', flush=True)
        walking = m.state is State.MOVE_TO_PICK
        m.trigger_move_pick()
        print('[slam-box] ' + ('walk over cancelled' if walking else 'walk over + pick requested'), flush=True)

    def walk(self, on):
        self.walking = on
        if not on:
            self.cmd_vel[:] = 0.0
            self.cmd_face[:] = 0.0

    # -- the stream ------------------------------------------------------------
    def _replan(self):
        """Cut at the frame the robot plays, put the root on the SLAM pelvis, roll
        the horizon forward, keep the matcher at the end of the first period."""
        m, mo = self.matcher, self.motion
        f = min(max(self.cur_frame, 1), mo.timesteps - 1)
        mo.truncate(f + 1)
        p = self.slam_pose()
        riding = m.state in (State.PICK, State.PLACE) or m.box_locked > 0
        if p is not None and not riding:
            xy, yaw = np.asarray(p['root_xy'], float), float(p['root_yaw'])
            self.sync_err = float(np.linalg.norm(xy - m.rootPos[0:2]))
            self.heading_err = math.degrees(wrap(yaw - m.rootYaw))
            jump = 0.0 if self.last_slam_xy is None else float(np.linalg.norm(xy - self.last_slam_xy))
            self.last_slam_xy = xy
            if jump > self.args.max_snap_jump:
                # SLAM jumped between two poses 0.2 s apart (the box fills the camera during
                # the lift): hold the reference root this time; the next pose is compared
                # against this one, so the snap resumes once SLAM holds still again
                self.snaps_skipped += 1
                print(f'[slam-box] SLAM pose jumped {jump:.2f} m in {self.args.replan_s:.1f} s: not snapping '
                      f'({self.snaps_skipped} skipped so far)', flush=True)
            else:
                seed_from_slam(m, xy, yaw)
        m.searchTimer = 0.0
        period = int(round(self.args.replan_s * POLICY_FPS))
        horizon = f + 1 + self.args.lookahead + period
        keep = None
        while mo.timesteps < horizon:
            mo.ensure(mo.timesteps + 1)
            if keep is None and mo.timesteps >= f + 1 + period:
                keep = (RM.snapshot(m), mo.time_mark())
        if keep is not None:
            RM.restore(m, keep[0])
            mo.time_restore(keep[1])
        self.pending_from = f + 1

    def tick(self, now):
        if self.t0 is None:
            self.t0 = now
        self.cur_frame = int((now - self.t0) * POLICY_FPS)
        if self.walking and not self.lost:
            yaw = self.matcher.rootYaw
            fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            self.cmd_vel, self.cmd_face = fwd * self.args.walk_speed, fwd.copy()
        if self.mode == MODE_POSE:
            if self.next_replan is None or now >= self.next_replan:
                self.next_replan = now + self.args.replan_s
                self._replan()
        else:
            self.next_replan = None
            self.motion.ensure(self.cur_frame + 1 + self.args.lookahead)
        newest = self.motion.timesteps - 1
        if newest > self.published or self.pending_from is not None:
            first = max(0, newest - self.args.lookahead)
            if self.pending_from is not None:
                first = min(first, self.pending_from)
                self.pending_from = None
            self.link.pose(self.motion, first, newest)
            self.published = newest
        if self.mode in (MODE_PLANNER, MODE_POSE) and now - self.last['heartbeat'] >= 1.0:
            self.link.command(planner=self.mode == MODE_PLANNER)
            self.last['heartbeat'] = now
        if self.mode == MODE_PLANNER and now - self.last['planner'] >= 0.1:
            self.last['planner'] = now
            self.link.planner_command(0, np.zeros(3), np.array([1.0, 0.0, 0.0]), -1.0)
        if now - self.last['bus'] >= 0.1:
            self.last['bus'] = now
            if self.mode != MODE_POSE:
                self.slam_pose()
            self._publish_state()
        if self.palm is not None and (self.last['arm_tau'] is None or now - self.last['arm_tau'] >= 0.02):
            self._send_palm_force(now)
        self._judge_grip(now)
        return newest

    def _send_palm_force(self, now):
        """--palm-force at 50 Hz: each hand's force ramps in while its contact label is on at
        the frame the robot plays (POSE only, not once the box is lost), and the torques at
        the measured joints go to the node (zeros while off; none at all without the flag)."""
        dt = 0.0 if self.last['arm_tau'] is None else now - self.last['arm_tau']
        self.last['arm_tau'] = now
        cur = max(0, min(self.cur_frame, self.motion.timesteps - 1))
        labels = self.motion.meta_at(cur)[4]
        on = self.mode == MODE_POSE and not self.lost
        step = dt / PalmForce.RAMP_S
        for h in range(2):
            r = self.palm_ramp[h]
            self.palm_ramp[h] = min(1.0, r + step) if on and labels[h] else max(0.0, r - step)
        state = self.robot.get()
        self.palm_tau = (np.zeros(14) if state is None or not any(self.palm_ramp)
                         else self.palm.torques(state[1], self.palm_ramp))
        self.link.arm_tau(self.palm_tau)

    def _judge_grip(self, now):
        """Once, after m's pick: an object on the floor within 1 m of the robot for most
        of --grip-check-s means the box was not lifted (or was pushed along)."""
        m = self.matcher
        if self.grip is None:
            if m.state is State.CARRY and not m.pick_at_stance and not getattr(self, 'grip_done', False):
                # b's pick plays without the box pose, so it is not judged either: b puts it down
                self.grip_done, self.grip_result = True, 'not checked'
                print('[slam-box] picked right here: no grip check (b puts it down)', flush=True)
                return
            if m.state is State.CARRY and m.box_locked == 0 and not getattr(self, 'grip_done', False):
                # the window opens --grip-check-delay after the pick; only a box MEASURED after
                # this moment counts (Boxer keeps the lifted box's track at its old floor spot)
                self.grip = [now + self.args.grip_check_delay, 0, 0]
                self.pick_spot = m.boxPos[0:2].copy() if not m.box_held else self._pick_spot
                objs = self.objects() or []
                self.grip_stamp = max([int(o.get('bbox_stamp_ns', 0)) for o in objs] + [0])
            return
        t_start, seen, asked = self.grip
        if now < t_start:
            return
        if now - t_start > self.args.grip_check_s:
            self.grip_done, self.grip = True, None
            self.lost = asked >= 2 and seen >= 0.5 * asked
            self.grip_result = 'lost' if self.lost else 'held'
            if self.lost:
                self.walk(False)
                print(f'[slam-box] grip check: an object stayed on the floor at the pick spot '
                      f'({seen} of {asked} looks): the box was lost, standing still', flush=True)
            else:
                print(f'[slam-box] grip check: nothing left on the floor at the pick spot '
                      f'({asked} looks): the box is held', flush=True)
            return
        if now - getattr(self, '_grip_t', 0.0) < 0.2:
            return
        self._grip_t = now
        objs = self.objects()
        if objs is None:
            return
        root = self.matcher.rootPos[0:2]
        rest = self.box.get('bottom') or 0.0               # where the box's bottom rested (a stand counts)
        # a box on the floor near the robot, measured after the pick (the sim's truth objects carry
        # no measurement stamp and always count)
        on_floor = any(np.hypot(*(np.asarray(o['bbox_center_world'][:2]) - root)) < 1.0
                       and o.get('z_range', [1.0])[0] < rest + 0.12 and o.get('track_state', 'active') != 'inactive'
                       and int(o.get('bbox_stamp_ns', self.grip_stamp + 1)) > self.grip_stamp
                       for o in objs)
        self.grip = [t_start, seen + int(on_floor), asked + 1]

    def _publish_state(self):
        m, mo = self.matcher, self.motion
        cur = max(0, min(self.cur_frame, mo.timesteps - 1))
        q = mo.qpos
        _, _, state_now, held_now, _ = mo.meta_at(cur) if mo.timesteps else (0, 0, m.state, False, (0, 0))
        # the reference's box in the reference's pelvis frame at the played frame: where
        # the clip holds it, for a sim that pins the box kinematically (run_sim.py --box-grasp kinematic)
        f = q[cur]
        box_local = [quat.inv_mul_vec(f[3:7], f[36:39] - f[0:3]).round(4).tolist(),
                     quat.mul(quat.inv(f[3:7]), f[39:43]).round(5).tolist()]
        self.bus.publish(STATE_TOPIC, {
            'mode': self.mode, 'root_xy': m.rootPos[0:2].tolist(), 'root_yaw': float(m.rootYaw),
            'tracked_xy': q[cur, 0:2].tolist(), 'ahead': q[cur:self.published + 1:5, 0:2].round(3).tolist(),
            'grid': [0.0] * GRID_DIM, 'grid_source': None, 'scan_xy': None, 'scan_yaw': None,
            'map_ok': False, 'localized': self.slam is not None,
            'sync_err_m': self.sync_err, 'heading_err_deg': self.heading_err,
            'stick': self.cmd_vel[0:2].tolist(), 'max_speed': 1.0, 'route': None,
            'task': {'state': m.state.name, 'phase': self.phase, 'box_xy': m.boxPos[0:2].round(3).tolist(),
                     'box_yaw': yaw_of(m.boxRot), 'box_held': bool(m.box_held), 'fixed': self.box['fixed'],
                     'source': self.box['source'], 'size': self.box['size'], 'lost': self.lost,
                     'carrying': m.state is State.CARRY,
                     # at the frame the robot plays now (the matcher runs about a second ahead)
                     'state_now': state_now.name, 'held_now': bool(held_now), 'box_in_pelvis_now': box_local}})

    def close(self):
        self.link.close()
        self.robot.close()


# ---------------------------------------------------------------------- modes

def run_headless(args):
    br = SlamBoxBridge(args)
    br._pick_spot = br.matcher.boxPos[0:2].copy()
    time.sleep(1.0)                                       # PUB slow joiner
    t0 = time.monotonic()
    started = posed = triggered = carried = placed = False
    t_start = t_pose = t_carry = t_place = 0.0
    fix_tries = 0
    try:
        while time.monotonic() - t0 < args.headless:
            now = time.monotonic()
            if not started:
                ready = br.robot.get() is not None
                if args.start_after_anchor and ready:
                    st = br.status() if int(now * 2) != int((now - 0.004) * 2) else None
                    if not (st and st['anchor']['anchored']):
                        ready = False
                        if st and int(now) % 5 == 0 and int(now) != int(now - 0.5):
                            print(f"[headless] waiting to start: {st['anchor'].get('wait') or 'anchoring'}", flush=True)
                br.phase = 'waiting for the node and the anchor'
                if args.auto_start and ready:
                    for _ in range(3):
                        br.start()
                        time.sleep(0.1)
                    started, t_start = True, now
                    br.phase = 'standing (settle)'
            elif not br.box['fixed'] and not args.pick_here and now - t_start > args.settle:
                br.phase = 'fixing the box from the map server'
                if int(now * 2) != int((now - 0.004) * 2):
                    fix_tries += 1
                    if br.fix_box():
                        br._pick_spot = br.matcher.boxPos[0:2].copy()
                        br.phase = 'box fixed, standing until POSE'
                    elif now - t_start - args.settle > args.fix_timeout:
                        print(f'[headless] no box reported within {args.fix_timeout:.0f} s: keeping the belief '
                              f'{C.BOX_SPAWN_FWD:.2f} m ahead', flush=True)
                        br.box.update(fixed=True, source='belief')
            elif not posed and now - t_start >= args.pose_after:
                br.toggle_pose()
                posed, t_pose = True, now
                br.phase = 'POSE, about to walk over'
            elif posed and not triggered and now - t_pose > args.walk_after:
                if args.pick_here:
                    br.box_action()
                    br.phase = 'picking right here'
                else:
                    br.move_pick()
                    br.phase = 'walking over and picking'
                triggered = True
            elif triggered and not carried and getattr(br, 'grip_done', False):
                if br.lost:
                    br.phase = 'lost the box: standing'
                    carried = placed = True
                else:
                    br.walk(True)
                    carried, t_carry = True, now
                    br.phase = f'carrying {args.carry_s:.0f} s forward'
            elif carried and not placed and not br.lost and now - t_carry > args.carry_s:
                br.walk(False)
                if now - t_carry > args.carry_s + 2.0:
                    br.box_action()
                    placed, t_place = True, now
                    br.phase = 'placing'
            elif placed and not br.lost and br.matcher.state is State.LOCOMOTION and now - t_place > 2.0:
                br.phase = 'done: standing'
                if not getattr(br, 'done_said', False):
                    br.done_said = True
                    print('[headless] sequence done: the box is placed, standing', flush=True)
            br.tick(now)
            time.sleep(0.004)
        if args.auto_stop:
            br.stop()
    except KeyboardInterrupt:
        pass
    finally:
        br.close()


def run_terminal(args):
    br = SlamBoxBridge(args)
    br._pick_spot = br.matcher.boxPos[0:2].copy()
    print(__doc__.split('Terminal keys')[1].split('Headless')[0], flush=True)
    keys = []
    lock = threading.Lock()

    def reader():
        while True:
            if select.select([sys.stdin], [], [], 0.2)[0]:
                line = sys.stdin.readline()
                if not line:
                    return
                with lock:
                    keys.append(line.strip().lower())

    threading.Thread(target=reader, daemon=True).start()
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
                    br.request_fix()
                elif k == 'b':
                    br.box_action()
                elif k == 'm':
                    br.move_pick()
                elif k == 'w':
                    br.walk(not br.walking)
                    print(f'[slam-box] walking {"on" if br.walking else "off"}', flush=True)
                elif k == 'o':
                    br.stop()
                elif k == 'q':
                    return
            br.tick(time.monotonic())
            time.sleep(0.004)
    except KeyboardInterrupt:
        pass
    finally:
        br.close()


FOLLOW_S = 0.3                  # s: the window camera eases toward the robot (SLAM jitter, head bob)
CONTACT_RGBA = np.array([0.25, 0.9, 0.35, 1.0])


def run_window(args):
    import glfw
    import mujoco
    from mm_g1.viewer import InteractiveViewer, draw_gizmos
    from run_hardware import GHOST_RGBA

    br = SlamBoxBridge(args)
    br._pick_spot = br.matcher.boxPos[0:2].copy()
    # no map server (or a dead one): each unanswered request blocks the window for its 60 ms
    # timeout, some 15 a second; back off 1 s after one instead
    br.map.retry_s = 1.0
    model = mujoco.MjModel.from_xml_path(C.SCENE_BOX_SCENEBOT_XML if C.SCENEBOT_PICK else C.SCENE_BOX_XML)
    if C.SCENEBOT_PICK:
        model.geom('box_geom').size[:] = C.BOX_HALF
    data = mujoco.MjData(model)
    eye3 = np.eye(3).ravel()
    keys_l = ']\nP\nX\nB\nM\nW A S D\nArrows\nShift\nF\nT\nEnter\nO\nEsc'
    keys_r = ('start control (planner mode, the robot stands)\ntoggle planner <-> POSE\n'
              'fix the box from the map server (for M)\npick right here / put down while holding\n'
              'walk over to the fixed box + pick (M again cancels)\nmove (POSE only)\nface\nhold to run\n'
              'WASD frame heading/camera\ntoggle gizmos\nre-send the mode command\nstop -> damping (final)\nquit')

    class Window(InteractiveViewer):
        def __init__(self):
            super().__init__(model, data, br.matcher, width=1400, height=900,
                             title='G1 box bridge (SLAM) -- ] start, P mode, X fix, B / M pick, O stop')
            self.ctx.free()                      # the base class uses fontscale 150
            self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
            self.cam.azimuth, self.cam.elevation, self.cam.distance = 150.0, -25.0, 4.5
            self.heading_frame = args.frame == 'heading'
            self.gdata = mujoco.MjData(model)    # the stick figure: FK only
            self.ghost_bodies = [b for b in range(1, model.nbody)
                                 if 'box' not in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or '')]
            feet = {model.body(n).id for n in ('left_ankle_roll_link', 'right_ankle_roll_link')}
            self.foot_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in feet]
            self.corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
            self.slam = {'xy': None, 'yaw': 0.0, 't': 0.0}
            self.follow = {'xy': None, 't': 0.0}
            self.objs = {'list': [], 't': 0.0, 'near': None}
            self.actions = {glfw.KEY_RIGHT_BRACKET: br.start, glfw.KEY_P: br.toggle_pose, glfw.KEY_O: br.stop,
                            glfw.KEY_X: br.request_fix, glfw.KEY_B: br.box_action, glfw.KEY_M: br.move_pick,
                            glfw.KEY_ENTER: self._resend, glfw.KEY_F: self._toggle_frame}

        def _on_key(self, window, key, scancode, action, mods):
            if action == glfw.PRESS and key in self.actions:
                self.actions[key]()
            elif key != glfw.KEY_SPACE:          # the base class's Space resets the matcher: no reset here
                super()._on_key(window, key, scancode, action, mods)

        def _resend(self):
            if br.mode in (MODE_PLANNER, MODE_POSE):
                br.link.command(planner=br.mode == MODE_PLANNER)
                print(f'[slam-box] mode command re-sent (planner={int(br.mode == MODE_PLANNER)})', flush=True)

        def _toggle_frame(self):
            self.heading_frame = not self.heading_frame
            print(f'[slam-box] WASD frame -> {"heading" if self.heading_frame else "camera"}', flush=True)

        def _stick(self):
            """The held keys as the matcher's command, in POSE only: in idle and planner mode
            the robot stands and the reference is not snapped to it, so the keys would walk
            and turn the reference away from the robot until POSE. Nothing once the box is lost."""
            self._speed = 0.0
            if br.mode != MODE_POSE or br.lost:
                return np.zeros(3), np.zeros(3)
            saved = self.cam.azimuth
            if self.heading_frame:
                self.cam.azimuth = math.degrees(br.matcher.rootYaw)
            vel, face = self._command()
            self.cam.azimuth = saved
            n = float(np.linalg.norm(vel))
            if n > 1e-6 and br.matcher.state is State.CARRY:
                vel = vel / n * args.walk_speed          # the carry data's full stick
            self._speed = float(np.linalg.norm(vel))
            return vel, face

        def _geom(self, kind, size, pos, rgba, mat=eye3):
            if self.scene.ngeom >= self.scene.maxgeom:
                return None
            g = self.scene.geoms[self.scene.ngeom]
            mujoco.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float),
                                np.asarray(mat, float), np.asarray(rgba, np.float32))
            self.scene.ngeom += 1
            return g

        def _pose_robot(self):
            """The measured robot at its SLAM pelvis pose, never at the matcher's root (that
            one runs ahead of the robot). The map server is asked at SLAM's 30 Hz: the bridge
            refreshes its pose once per re-plan in POSE. Hidden without g1_debug or a pose."""
            now = time.monotonic()
            if now >= self.slam['t']:
                r = br.map.request({'op': 'pose'})
                p = None if r is None else r.get('pose')
                self.slam['t'] = now + 1.0 / 30
                if p is not None and p.get('age_s', 9.0) < 0.5:
                    self.slam.update(xy=np.asarray(p['root_xy'], float), yaw=float(p['root_yaw']))
            data.qpos[0:7] = (0.0, 0.0, -5.0, 1.0, 0.0, 0.0, 0.0)
            state = br.robot.get()
            if state is None or self.slam['xy'] is None:
                return False
            bq, q = state
            data.qpos[3:7] = quat.mul(yaw_quat(self.slam['yaw'] - yaw_of(bq)), bq)
            data.qpos[7:36] = q
            data.qpos[0:3] = (self.slam['xy'][0], self.slam['xy'][1], 1.0)
            mujoco.mj_kinematics(model, data)
            low = min(float((data.geom_xpos[g] + (model.geom_aabb[g, :3] + self.corners * model.geom_aabb[g, 3:])
                             @ data.geom_xmat[g].reshape(3, 3).T)[:, 2].min()) for g in self.foot_geoms)
            data.qpos[2] -= low - 0.002
            return True

        def _tint_box(self, cur):
            """Green while the played frame's hand labels are on; faint while the box pose
            is the belief (not fixed, not held)."""
            if self._box_gid is None:
                return
            _, _, _, held, (lc, rc) = br.motion.meta_at(cur)
            rgba = self._box_rgba.copy()
            if lc or rc:
                rgba = CONTACT_RGBA.copy()
            elif not br.box['fixed'] and not held:
                rgba[3] = 0.3
            model.geom_rgba[self._box_gid] = rgba

        def _draw_objects(self):
            """What the map server reports (Boxer's boxes, or the sim's truth), as in UI 1's
            Boxer objects card: the one x would take in amber, the others blue, and until
            the fix the disc x looks in (--box-radius round the belief)."""
            now = time.monotonic()
            if now - self.objs['t'] > 0.5:
                self.objs['t'] = now
                got = br.objects()
                if got is not None:
                    self.objs['list'] = got
            belief = np.array([C.BOX_SPAWN_FWD, C.BOX_SPAWN_LAT])
            pick = None
            if not br.box['fixed']:
                pick = pick_object(self.objs['list'], belief, args.box_radius)
                self._geom(mujoco.mjtGeom.mjGEOM_CYLINDER, (args.box_radius, 0.003, 0.0),
                           (belief[0], belief[1], 0.003), (0.95, 0.75, 0.2, 0.12))
            self.objs['near'] = None if pick is None else \
                float(np.hypot(*(np.asarray(pick['bbox_center_world'][:2], float) - belief)))
            for o in self.objs['list']:
                T = np.asarray(o['T_world_object'], float).reshape(4, 4)
                rgba = [0.95, 0.7, 0.15, 0.5] if o is pick else [0.3, 0.55, 0.95, 0.35]
                if o.get('track_state', 'active') == 'inactive':
                    rgba[3] = 0.12
                self._geom(mujoco.mjtGeom.mjGEOM_BOX, 0.5 * np.asarray(o['bbox_size_xyz'], float),
                           o['bbox_center_world'], rgba, T[:3, :3].ravel())

        def _draw_reference(self, cur):
            g = self.gdata
            g.qpos[0:36] = br.motion.sent_qpos[cur][0:36]     # with the squeeze and pitch biases, as streamed
            mujoco.mj_kinematics(model, g)
            for b in self.ghost_bodies:
                pa = model.body_parentid[b]
                a, c = g.xpos[pa], g.xpos[b]
                if pa == 0 or np.linalg.norm(c - a) < 1e-6:
                    continue
                gm = self._geom(mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), GHOST_RGBA[br.mode])
                if gm is not None:
                    mujoco.mjv_connector(gm, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.025,
                                         np.asarray(a, float), np.asarray(c, float))

        def _draw_gizmos(self, q):
            # the command trajectory is drawn at the matcher's root, which runs ahead of the
            # played frame: shift it (two geoms a point) onto the stick figure; m's route and
            # stance stay where they are, in the map frame
            n0 = self.scene.ngeom
            draw_gizmos(self.scene, br.matcher)
            dxy = q[0:2] - br.matcher.rootPos[0:2]
            for i in range(n0, min(n0 + 2 * len(br.matcher.Tpos), self.scene.ngeom)):
                self.scene.geoms[i].pos[0] += dxy[0]
                self.scene.geoms[i].pos[1] += dxy[1]

        def _status(self, cur, live):
            m, box = br.matcher, br.box
            _, _, state_now, held_now, _ = br.motion.meta_at(cur)
            mode = {MODE_IDLE: 'IDLE       next: ] (planner mode, the robot stands)',
                    MODE_PLANNER: 'PLANNER    the robot stands; next: P -> POSE',
                    MODE_POSE: 'POSE       the box matcher drives the robot',
                    MODE_STOPPED: 'STOPPED    damping; restart deploy.sh to resume'}[br.mode]
            task = f'{state_now.name}{"  (holding)" if held_now else ""}   the matcher: {m.state.name}'
            if self._speed > 0:
                task += f'   {self._speed:.2f} m/s'
            if box['fixed'] and box.get('xy') is not None:
                sx, sy, sz = box['size']
                boxl = (f"fixed at ({box['xy'][0]:+.2f}, {box['xy'][1]:+.2f}) m, "
                        f"yaw {math.degrees(box['detected_yaw']):+.0f} deg, {sx:.2f} x {sy:.2f} x {sz:.2f} m, "
                        f"from {box['source']}")
            elif box['fixed']:
                boxl = f"the belief, {C.BOX_SPAWN_FWD:.2f} m ahead ({box['source']})"
            else:
                boxl = (f'not fixed: the belief {C.BOX_SPAWN_FWD:.2f} m ahead, {C.BOX_SPAWN_LAT:+.2f} m left '
                        f'(X fixes it, for M)')
            n = len(self.objs['list'])
            objl = (f'{n} reported' + ('' if box['fixed'] else
                    (f', {self.objs["near"]:.2f} m from the belief' if self.objs['near'] is not None
                     else f', none within {args.box_radius:.1f} m of the belief')))
            grip = {'lost': 'LOST: standing, B and M ignored', 'held': 'held (the check saw no box left on the floor)',
                    'not checked': 'not checked (B picked right here)'}.get(br.grip_result,
                                                                          'checking...' if br.grip else '-')
            loc = br.slam
            locl = (f"({loc['root_xy'][0]:+.2f}, {loc['root_xy'][1]:+.2f}) "
                    f"{math.degrees(loc['root_yaw']):+.0f} deg  [{loc.get('link') or '?'}]" if loc else 'NOT LOCALIZED')
            sync = ('-' if br.sync_err is None else
                    f'{br.sync_err:.2f} m, heading {br.heading_err:+.1f} deg (before each snap)'
                    + (f', {br.snaps_skipped} jumps not snapped' if br.snaps_skipped else ''))
            robot = 'LIVE' if live else ('no g1_debug' if br.robot.get() is None else 'NOT LOCALIZED')
            palm = ('off (--palm-force N)' if br.palm is None else
                    f'{br.palm.force:.0f} N   left {br.palm_ramp[0]:.0%}  right {br.palm_ramp[1]:.0%}   '
                    f'max {np.abs(br.palm_tau).max():.1f} Nm, sent as q_target + tau / kp')
            return ('mode\ntask\nbox\nobjects\ngrip\npalm force\nrobot (SLAM)\nref vs robot\nrobot state\nlink',
                    f'{mode}\n{task}\n{boxl}\n{objl}\n{grip}\n{palm}\n{locl}\n{sync}\n{robot}\n{br.link.endpoint}')

        def run(self, max_frames=None):
            frames = 0
            try:
                while not glfw.window_should_close(self.window):
                    if max_frames is not None and frames >= max_frames:
                        break
                    frames += 1
                    br.cmd_vel, br.cmd_face = self._stick()
                    br.tick(time.monotonic())
                    cur = max(0, min(br.cur_frame, br.motion.timesteps - 1))
                    q = br.motion.qpos[cur]
                    live = self._pose_robot()
                    data.qpos[36:43] = q[36:43]          # the box where the played frame has it
                    self._tint_box(cur)
                    mujoco.mj_forward(model, data)
                    # follow the robot (the reference without it), eased so SLAM jitter does not shake the view
                    target, t = (self.slam['xy'] if live else q[0:2]), time.monotonic()
                    if self.follow['xy'] is None:
                        self.follow['xy'] = np.array(target, float)
                    else:
                        k = 1.0 - math.exp(-(t - self.follow['t']) / FOLLOW_S)
                        self.follow['xy'] += k * (target - self.follow['xy'])
                    self.follow['t'] = t
                    self.cam.lookat[0], self.cam.lookat[1] = float(self.follow['xy'][0]), float(self.follow['xy'][1])

                    w, h = glfw.get_framebuffer_size(self.window)
                    vp = mujoco.MjrRect(0, 0, w, h)
                    mujoco.mjv_updateScene(model, data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL,
                                           self.scene)
                    self._draw_objects()
                    self._draw_reference(cur)
                    if self.show_traj:
                        self._draw_gizmos(q)
                    mujoco.mjr_render(vp, self.scene, self.ctx)
                    labels, values = self._status(cur, live)
                    mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, vp,
                                       labels, values, self.ctx)
                    mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, vp,
                                       keys_l, keys_r, self.ctx)
                    if br.lost:
                        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOP, vp,
                                           'BOX LOST: standing still', None, self.ctx)
                    glfw.swap_buffers(self.window)
                    glfw.poll_events()
            except KeyboardInterrupt:
                pass
            finally:
                br.close()
                glfw.terminate()

    Window().run(max_frames=args.smoke_frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bind', default='*')
    ap.add_argument('--port', type=int, default=5556)
    ap.add_argument('--robot-host', default='localhost')
    ap.add_argument('--robot-port', type=int, default=5557)
    ap.add_argument('--map-rep', default='tcp://127.0.0.1:5591', help="live.server's REP")
    ap.add_argument('--bus-pub', default=wire.BUS_PUB)
    ap.add_argument('--lookahead', type=int, default=50, help='50 Hz frames the stream runs ahead')
    ap.add_argument('--replan-s', type=float, default=0.2, help='s between cuts that re-anchor the root to SLAM')
    ap.add_argument('--max-snap-jump', type=float, default=0.5,
                    help='m: a SLAM pose this far from the previous one (0.2 s earlier) is a jump, not a step '
                         '(the box fills the camera during the lift): the reference root is held that time')
    ap.add_argument('--box-fwd', type=float, default=2.0, help='box belief: m ahead of the start pose')
    ap.add_argument('--box-lat', type=float, default=0.0, help='box belief: m to the left of the start pose')
    ap.add_argument('--box-radius', type=float, default=1.0, help='m around the belief a reported object may be')
    ap.add_argument('--fix-range-bias', type=float, default=0.0, metavar='M',
                    help="m the fixed box is pushed away from the robot: Boxer's centre sits toward the camera "
                         '(4 to 5 cm at 2 m in the sim; measure it on the robot)')
    ap.add_argument('--walk-speed', type=float, default=C.CARRY_MAX_SPEED,
                    help='m/s while carrying forward (default the full stick of the carry data, 0.75: '
                         'slower commands match its near-stationary frames and the robot stands)')
    ap.add_argument('--grip-check-s', type=float, default=1.5)
    ap.add_argument('--grip-check-delay', type=float, default=4.0,
                    help="s after the pick before the grip check: Boxer's track of the box on the floor outlives the lift")
    ap.add_argument('--shoulder-squeeze', type=float, default=0.0,
                    help='rad of inward shoulder roll in the reference while a hand contact label is on')
    ap.add_argument('--wrist-squeeze', type=float, default=0.0, help='rad of inward wrist yaw, the same way')
    ap.add_argument('--palm-force', type=float, default=0.0, metavar='N',
                    help='newtons pressing each palm toward the other while its contact label is on, through the '
                         "arm's Jacobian (tau = J^T f at the measured joints), sent to the node on arm_tau, which adds "
                         "tau / kp to SONIC's arm targets. Off by default; with a box add --palm-force 40 (the sim: "
                         '40 N lifted, held and placed the SceneBot box, 25 N tipped it, 15 N dropped it)')
    ap.add_argument('--fwd-tol', type=float, default=None, metavar='M',
                    help='the pick fires only with the root this close to the stance along the approach rail '
                         '(short of it the servo keeps creeping). Off by default: at 0.06 the robot hovered round '
                         'the stance and the pick timed out; a lateral miss is absorbed by the hands anyway')
    ap.add_argument('--arrive-near', type=float, default=None, metavar='M',
                    help='tighten the pick entry: fire within this radius of the stance (default the data\'s 0.12 m, '
                         'loose fallback 0.30 m after 1 s)')
    ap.add_argument('--approach-scale', type=float, default=0.5,
                    help='scale on the walk-over speeds: the real robot overshoots a fast reference stop into the box')
    ap.add_argument('--stance-bias', type=float, default=-0.05,
                    help='m added to the box-ahead-of-stance offset of the data (negative: stand closer), '
                         'to absorb the drift back from the reference during the squat')
    ap.add_argument('--shoulder-pitch-bias', type=float, default=0.3,
                    help='rad added to both shoulder pitches during a pick or place ride (positive swings '
                         'the reaching arms down: the robot squats shallower than the clip)')
    ap.add_argument('--elbow-bias', type=float, default=0.0,
                    help='rad added to both elbows during a pick or place ride (negative straightens the arms: '
                         'the hands reach farther forward and down)')
    ap.add_argument('--no-center-hands', action='store_true',
                    help="do not shift the arms sideways by the box's lateral offset at the pick entry")
    ap.add_argument('--no-hold-arms', action='store_true',
                    help="let the carry search move the arms after the pick (default: the arms stay in the pick "
                         "clip's final pose until the place, the single SceneBot motion)")
    ap.add_argument('--shoulder-open', type=float, default=0.0,
                    help='rad of outward shoulder roll during a pick or place ride while both labels are off')
    ap.add_argument('--headless', type=float, default=None, metavar='SECONDS')
    ap.add_argument('--auto-start', action='store_true', help='headless: ] once the node is up (moves the robot!)')
    ap.add_argument('--start-after-anchor', action='store_true', help='headless: ] only once the map server is anchored')
    ap.add_argument('--settle', type=float, default=5.0, help='headless: s standing before the box fix')
    ap.add_argument('--fix-timeout', type=float, default=90.0,
                    help='headless: s to wait for a reported box (Boxer loads its models for some 40 s)')
    ap.add_argument('--pose-after', type=float, default=12.0, help='headless: s after start, P -> POSE')
    ap.add_argument('--walk-after', type=float, default=3.0, help='headless: s in POSE before the box action')
    ap.add_argument('--pick-here', action='store_true',
                    help='headless: pick right where the robot stands (b) instead of walking over first (m), '
                         'with no box fix')
    ap.add_argument('--carry-s', type=float, default=0.0,
                    help='headless: s walking forward with the box before the place (default 0: the single SceneBot '
                         'motion, pick then hold then place, which holds the box; the carry search drops it)')
    ap.add_argument('--auto-stop', action='store_true')
    ap.add_argument('--terminal', action='store_true', help='terminal keys (letter + Enter) instead of the window')
    ap.add_argument('--frame', choices=('heading', 'camera'), default='heading',
                    help="window: WASD relative to the reference's heading or to the view (F toggles)")
    ap.add_argument('--smoke-frames', type=int, default=None, help='(testing) close the window after N frames')
    args = ap.parse_args()
    if args.headless is not None:
        run_headless(args)
    elif args.terminal:
        run_terminal(args)
    else:
        run_window(args)


if __name__ == '__main__':
    main()
