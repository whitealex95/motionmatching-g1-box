"""The box task (walk over, pick, carry, place) on the real G1 with SLAM
localization from motionmatching-g1-loco's live stack.

The scenario: the box stands about 2 m in front of the start pose. SLAM is
initialized and the map server anchors its frame at the start pose (that frame
is this matcher's world: origin at the start pose, facing +x). Nothing is
mapped. Once, after the anchor, the box pose is fixed from what the map server
reports (`objects`: Boxer's boxes in the map frame, or the sim's true box with
`live.server --objects truth`). From then on the box relative to the robot
follows from that fixed pose and the SLAM pose, so the lift needs no camera.
Every --replan-s the stream is cut at the frame the robot plays and the
matcher's root is put on the SLAM pelvis (not during the ridden pick and place:
those play open loop from the committed pose). After the pick the grip is
judged once from what the server still sees on the floor at the pick spot:
lost means stand still, no second try.

    python sonic_g1_box/run_slam.py                       # terminal keys
    python sonic_g1_box/run_slam.py --headless 240 --auto-start --start-after-anchor

Terminal keys (type the letter, then Enter):
  ]  start (planner mode, the robot stands)    p  planner <-> POSE
  x  fix the box from the map server           b  box action (walk over + pick, or place)
  w  walk forward (--walk-speed) / stop        o  stop (damping, final)       q  quit

Headless: waits for the node and the anchor, starts, settles, fixes the box,
enters POSE, triggers the pick, judges the grip, carries --carry-s seconds
forward, places, stands.
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
from mm_stream import MMMotion, POLICY_FPS
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
        self.held_arms = None                  # the pick clip's final arm pose, kept through the carry
        self.center_hands = center_hands
        self.center_roll = 0.0                 # both shoulder rolls, from the box's lateral offset at the pick entry
        self._last_state = None
        super().__init__(matcher, commander)

    def _step_matcher(self):
        m = self.matcher
        before = m.state
        super()._step_matcher()
        if self.center_hands and m.state is State.PICK and before is not State.PICK:
            # the robot stops beside the stance, not on it: the box's lateral offset from the
            # pelvis at this moment (the SLAM pose against the fixed box) becomes a roll on
            # both shoulders, so the clamp closes where the box is
            left = float(quat.inv_mul_vec(m.rootRot, m.boxPos - m.rootPos)[1])
            self.center_roll = float(np.clip(left / self.ARM_LEVER, -0.5, 0.5))
            print(f'[slam-box] pick entry: the box is {left:+.2f} m left of the pelvis, '
                  f'shoulder rolls {self.center_roll:+.2f} rad to centre the hands', flush=True)
        elif m.state in (State.LOCOMOTION, State.MOVE_TO_PICK):
            self.center_roll = 0.0

    def _rebuild(self):
        super()._rebuild()
        sq, wq, op, sp, eb = self.bias
        if not self._meta or not (any(self.bias) or self.hold_arms):
            return
        q = self._qpos.copy()
        prev_state = None
        cr = self.center_roll
        for i, (_, _, state, held, (lc, rc)) in enumerate(self._meta[:len(q)]):
            if cr and (state in (State.PICK, State.PLACE) or (state is State.CARRY and held)):
                q[i, 7 + L_SHOULDER_ROLL] += cr      # both arms toward the box (+: left)
                q[i, 7 + R_SHOULDER_ROLL] += cr
            if self.hold_arms:
                # the single SceneBot motion: after its pick the carry search would swap
                # in other clips' arms and the box drops, so the arms stay where the pick
                # left them until the reverse clip (the place) takes over
                if state is State.CARRY and prev_state is State.PICK and self.held_arms is None:
                    self.held_arms = q[i - 1, ARMS].copy()
                if state is State.CARRY and self.held_arms is not None:
                    q[i, ARMS] = self.held_arms
                elif state is not State.CARRY:
                    self.held_arms = None if state is not State.PICK else self.held_arms
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
        from sonic_tracking import params as P
        angles = q[:, 7:36][:, P.MUJOCO_TO_ISAACLAB]
        speeds = np.zeros_like(angles)
        if len(angles) > 1:
            speeds[1:] = (angles[1:] - angles[:-1]) * POLICY_FPS
        self._joint_pos, self._joint_vel = angles, speeds


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

    def __init__(self, endpoint, timeout_ms=60):
        self.endpoint, self.timeout_ms = endpoint, timeout_ms
        self._open()

    def _open(self):
        self.sock = zmq.Context.instance().socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.connect(self.endpoint)

    def request(self, q):
        try:
            self.sock.send(msgpack.packb(q, use_bin_type=True))
            r = msgpack.unpackb(self.sock.recv(), raw=False)
            return r if r.get('ok') else None
        except zmq.ZMQError:
            self.sock.close(0)
            self._open()
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
            C.MOVE_END_SPEED_MIN *= k
            C.MOVE_END_SPEED_MAX *= k
            # the route aims MOVE_OVERSHOOT past the stance to keep the walk alive: with the
            # SceneBot box that is the box itself, and the robot runs into it. The slow walk
            # needs no overshoot, and the servo stops commanding earlier so the robot's lag
            # lands it on the stance
            C.MOVE_OVERSHOOT = 0.10
            C.MOVE_STOP_DIST = 0.30
            print(f'[slam-box] approach speeds x{k:.2f} (route up to {C.MOVE_ROUTE_SPEED_MAX:.2f} m/s, endgame '
                  f'{C.MOVE_END_SPEED_MIN:.2f} to {C.MOVE_END_SPEED_MAX:.2f} m/s, overshoot {C.MOVE_OVERSHOOT:.2f} m, '
                  f'commands stop {C.MOVE_STOP_DIST:.2f} m before the stance)', flush=True)
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
        self.phase = 'idle'
        self.last = {'heartbeat': -1.0, 'planner': -1.0, 'bus': -1.0}
        self.walking = False

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
        T = np.asarray(o['T_world_object'], float).reshape(4, 4)
        yaw = float(np.arctan2(T[1, 0], T[0, 0]))
        # the box's symmetry (BOX_ROT_FOLDS, 4 for the cube): the fold nearest the belief's
        # orientation, so the detected yaw does not send the approach round the side
        period = 2.0 * np.pi / max(int(getattr(C, 'BOX_ROT_FOLDS', 1)), 1)
        yaw = (yaw + 0.5 * period) % period - 0.5 * period
        size = [float(v) for v in o['bbox_size_xyz']]
        m.boxPos = np.array([c[0], c[1], C.BOX_REST_Z])
        m.boxRot = quat.mul(yaw_quat(yaw - yaw_of(m.box_spawn_rot)), m.box_spawn_rot)
        self.box.update(fixed=True, source=str(o.get('language_label') or 'object'), size=size,
                        detected_yaw=yaw, track_id=o.get('track_id'), bottom=float(c[2] - 0.5 * size[2]))
        lib_size = [2.0 * v for v in C.BOX_HALF]
        note = '' if max(abs(a - b) for a, b in zip(sorted(size), sorted(lib_size))) < 0.1 else \
            f' (the motion library is baked for a {lib_size[0]:.2f} x {lib_size[1]:.2f} x {lib_size[2]:.2f} m box)'
        print(f'[slam-box] box fixed at ({c[0]:.2f}, {c[1]:.2f}) m, yaw {math.degrees(yaw):.0f} deg, '
              f'size {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m, from {self.box["source"]}{note}', flush=True)
        return True

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
        if self.lost:
            print('[slam-box] the box was lost: standing, no second try', flush=True)
            return
        self.matcher.trigger_box()
        print(f'[slam-box] box action requested (state {self.matcher.state.name})', flush=True)

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
        self._judge_grip(now)
        return newest

    def _judge_grip(self, now):
        """Once, after the pick: an object on the floor within 1 m of the robot for most
        of --grip-check-s means the box was not lifted (or was pushed along)."""
        m = self.matcher
        if self.grip is None:
            if m.state is State.CARRY and m.box_locked == 0 and not getattr(self, 'grip_done', False):
                # the window opens --grip-check-delay after the pick: Boxer keeps a track of the
                # box on the floor for a few seconds after it was lifted
                self.grip = [now + self.args.grip_check_delay, 0, 0]
                self.pick_spot = m.boxPos[0:2].copy() if not m.box_held else self._pick_spot
            return
        t_start, seen, asked = self.grip
        if now < t_start:
            return
        if now - t_start > self.args.grip_check_s:
            self.grip_done, self.grip = True, None
            self.lost = asked >= 2 and seen >= 0.5 * asked
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
        on_floor = any(np.hypot(*(np.asarray(o['bbox_center_world'][:2]) - root)) < 1.0
                       and o.get('z_range', [1.0])[0] < rest + 0.12 and o.get('track_state', 'active') != 'inactive'
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
            elif not br.box['fixed'] and now - t_start > args.settle:
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
                br.box_action()
                triggered = True
                br.phase = 'walking over and picking'
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
                    if not br.fix_box():
                        print('[slam-box] no object near the belief: is Boxer running, is the map anchored?', flush=True)
                elif k == 'b':
                    br.box_action()
                elif k == 'n':
                    br.matcher.trigger_pick_instant()
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
    ap.add_argument('--walk-speed', type=float, default=C.CARRY_MAX_SPEED,
                    help='m/s while carrying forward (default the full stick of the carry data, 0.75: '
                         'slower commands match its near-stationary frames and the robot stands)')
    ap.add_argument('--grip-check-s', type=float, default=1.5)
    ap.add_argument('--grip-check-delay', type=float, default=4.0,
                    help="s after the pick before the grip check: Boxer's track of the box on the floor outlives the lift")
    ap.add_argument('--shoulder-squeeze', type=float, default=0.45,
                    help='rad of inward shoulder roll in the reference while a hand contact label is on')
    ap.add_argument('--wrist-squeeze', type=float, default=0.30, help='rad of inward wrist yaw, the same way')
    ap.add_argument('--arrive-near', type=float, default=None, metavar='M',
                    help='tighten the pick entry: fire within this radius of the stance (default the data\'s 0.12 m, '
                         'loose fallback 0.30 m after 1 s)')
    ap.add_argument('--approach-scale', type=float, default=0.5,
                    help='scale on the walk-over speeds: the real robot overshoots a fast reference stop into the box')
    ap.add_argument('--stance-bias', type=float, default=0.0,
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
    ap.add_argument('--carry-s', type=float, default=0.0,
                    help='headless: s walking forward with the box before the place (default 0: the single SceneBot '
                         'motion, pick then hold then place, which holds the box; the carry search drops it)')
    ap.add_argument('--auto-stop', action='store_true')
    args = ap.parse_args()
    if args.headless is not None:
        run_headless(args)
    else:
        run_terminal(args)


if __name__ == '__main__':
    main()
