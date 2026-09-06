"""Stream the box motion matcher to the REAL G1 through the C++ deploy node.

Plays the role the Pico manager plays in the lab's teleop stack: this process
binds a ZMQ PUB socket and the deploy node (`motionmatching-g1-deploy`,
`./deploy.sh --input-type zmq_manager real ...`) subscribes to it. The mode
sequence mirrors the upstream Pico flow:

  damping -> ] -> PLANNER (robot stands, IDLE planner msgs at 10 Hz)
          -> P -> POSE    (motion-matching pose stream drives the robot)
          -> P -> PLANNER (toggle back = soft pause)
          -> O -> damping (stop; not resumable)

  command topic   {start, stop, planner}     planner=1 -> planner mode,
                                             planner=0 -> POSE (streamed) mode
  planner topic   locomotion command at 10 Hz while in planner mode: WASD /
                  arrows steer it like the upstream Pico joystick (IDLE when no
                  key is held). The applied command is shown in the status
                  panel and printed to the terminal when it changes.
  pose topic      protocol v1: joint_pos / joint_vel [N,29] (IsaacLab order),
                  body_quat [N,4] (w,x,y,z), frame_index [N] int64, at 50 Hz

The window shows two characters, like run_interactive.py:
  - the full robot model = the ACTUAL robot, posed from the deploy node's
    `g1_debug` output (port 5557), grounded at the origin (the node publishes
    no odometry). Without data it stands still in the default pose, exactly
    like the run_sim.py spawn.
  - a stick figure = the kinematic target the matcher is streaming. Its color
    is the bridge mode: gray before any control (and after stop), blue in
    planner mode, orange in POSE (tracking). It starts pelvis-aligned with the
    robot; --ref-mode anchor-xy (off by default) additionally keeps pulling
    the reference's tracked frame toward the robot's xy (robot-centric view;
    display only, the v1 pose stream carries no root xy).

The matcher runs LOOKAHEAD frames ahead of wall-clock time, so every chunk
carries real future frames for the SONIC encoder (it looks 45 frames ahead).
The robot therefore reacts to a key press about LOOKAHEAD / 50 s later.

Window keys:

  W / A / S / D    move, relative to the reference's current heading (default;
                   press F or use --frame camera for view-relative)
  Arrow keys       face direction
  Shift (hold)     walk instead of run
  F                toggle the WASD frame: heading <-> camera
  B                box action: walk over + pick up / set down
  N                pick up NOW (skip the walk-over)
  T                toggle gizmos
  ]                START: enter PLANNER mode (upstream A+B+X+Y)
  P                toggle PLANNER <-> POSE   (upstream A+X)
  O                STOP: damping (not resumable)
  Enter            re-send the current mode command
  Space            reset the reference (stick figure + camera back to spawn).
                   Blocked in POSE mode: teleporting the tracked reference
                   would make the robot jump. Switch to planner mode first.
  Esc              quit this window (the deploy node keeps the last frames)

    python run_hardware.py                     # bind tcp://*:5556
    python run_hardware.py --frame body --lookahead 50
    python run_hardware.py --headless 10       # no window, idle reference, 10 s (smoke test)
"""
import argparse
import json
import math
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
os.environ.setdefault('MUJOCO_GL', 'glfw')

import numpy as np
import zmq

from mm_g1 import config as C
from mm_g1 import quat
from mm_g1.data import load_library
from mm_g1.controller import MotionMatcher
from mm_g1.features import yaw_quat
from mm_g1.states import State
from mm_stream import MMMotion, POLICY_FPS
import ref_modes as RM

# --- wire format (mirror of motionmatching-g1-deploy/tools/zmq_protocol.py) ---
HEADER_SIZE = 1280
_DTYPE_STR = {'float32': 'f32', 'float64': 'f64', 'int32': 'i32',
              'int64': 'i64', 'uint8': 'u8', 'bool': 'bool'}

MODE_IDLE = 'idle'          # before start: node in WAIT_FOR_CONTROL
MODE_PLANNER = 'planner'    # node in PLANNER mode, robot stands (IDLE)
MODE_POSE = 'pose'          # node in STREAMED_MOTION, tracks the pose stream
MODE_STOPPED = 'stopped'    # damping sent; deploy.sh must be restarted

ROBOT_STALE_S = 0.5         # g1_debug older than this counts as "no data"

# default standing pose, hardware order (policy_parameters.hpp default_angles);
# also what run_sim.py spawns in
DEFAULT_QJ = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,      # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,      # right leg
    0.0, 0.0, 0.0,                             # waist
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,         # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,        # right arm
])


def pack_message(topic, arrays, version):
    fields, blobs = [], []
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        if arr.dtype.byteorder == '>':
            arr = arr.astype(arr.dtype.newbyteorder('<'))
        fields.append({'name': name, 'dtype': _DTYPE_STR[arr.dtype.name],
                       'shape': list(arr.shape)})
        blobs.append(arr.tobytes())
    header = json.dumps({'v': version, 'endian': 'le', 'count': 1,
                         'fields': fields}).encode()
    if len(header) > HEADER_SIZE:
        raise ValueError(f'header too large: {len(header)} > {HEADER_SIZE}')
    return topic.encode() + header.ljust(HEADER_SIZE, b'\0') + b''.join(blobs)


class DeployLink:
    """PUB side of the deploy node's zmq_manager (command/planner/pose topics)."""

    def __init__(self, bind='*', port=5556):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 10)
        self.sock.bind(f'tcp://{bind}:{port}')
        self.endpoint = f'tcp://{bind}:{port}'
        self.n_pose = 0
        time.sleep(0.3)                    # let a waiting subscriber connect

    def command(self, start=False, stop=False, planner=False):
        u8 = lambda b: np.array([1 if b else 0], dtype=np.uint8)
        self.sock.send(pack_message('command', {
            'start': u8(start), 'stop': u8(stop), 'planner': u8(planner)},
            version=1))

    def planner_command(self, mode, movement, facing, speed=-1.0, height=-1.0):
        self.sock.send(pack_message('planner', {
            'mode': np.array([mode], dtype=np.int32),            # LocomotionMode enum
            'movement': np.asarray(movement, dtype=np.float32),  # (3,) unit dir
            'facing': np.asarray(facing, dtype=np.float32),      # (3,) unit dir
            'speed': np.array([speed], dtype=np.float32),        # m/s, -1 default
            'height': np.array([height], dtype=np.float32),
        }, version=1))

    def planner_idle(self):
        # same IDLE state the node seeds on entering planner mode
        self.planner_command(0, np.zeros(3), (1.0, 0.0, 0.0))

    def pose(self, stream, first, last):
        sl = slice(first, last + 1)
        self.sock.send(pack_message('pose', {
            'joint_pos': stream.joint_pos[sl].astype(np.float32),      # (N, 29) IsaacLab
            'joint_vel': stream.joint_vel[sl].astype(np.float32),      # (N, 29)
            'body_quat': stream.qpos[sl, 3:7].astype(np.float32),      # (N, 4) w,x,y,z
            'frame_index': np.arange(first, last + 1, dtype=np.int64),  # (N,)
        }, version=1))
        self.n_pose += 1

    def close(self):
        self.sock.close(linger=200)
        self.ctx.term()


class RobotState:
    """SUB side of the deploy node's g1_debug output (port 5557, msgpack)."""

    def __init__(self, host='localhost', port=5557, topic='g1_debug'):
        import msgpack
        self._unpackb = msgpack.unpackb
        self.topic = topic.encode()
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, 10)
        self.sock.setsockopt(zmq.SUBSCRIBE, self.topic)
        self.sock.connect(f'tcp://{host}:{port}')
        self.lock = threading.Lock()
        self.base_quat = None        # (4,) w,x,y,z
        self.body_q = None           # (29,) MuJoCo order
        self.stamp = 0.0
        self._stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        while not self._stop:
            if not poller.poll(100):
                continue
            msg = self.sock.recv()
            try:
                d = self._unpackb(msg[len(self.topic):], raw=False)
                quat = np.asarray(d['base_quat'], dtype=np.float64)      # (4,)
                q = np.asarray(d['body_q'], dtype=np.float64)            # (29,)
            except Exception:
                continue
            with self.lock:
                self.base_quat, self.body_q = quat, q
                self.stamp = time.monotonic()

    def get(self):
        """Returns (base_quat (4,), body_q (29,)) or None when stale/absent."""
        with self.lock:
            if self.body_q is None or time.monotonic() - self.stamp > ROBOT_STALE_S:
                return None
            return self.base_quat.copy(), self.body_q.copy()

    def close(self):
        self._stop = True
        self.thread.join(timeout=1.0)
        self.sock.close(linger=0)
        self.ctx.term()


LOCO_NAMES = {0: 'IDLE', 1: 'SLOW_WALK', 2: 'WALK', 3: 'RUN'}


class Streamer:
    """Real-time driver: keeps the matcher LOOKAHEAD frames ahead of the clock
    and publishes the window [newest - lookahead, newest] whenever it grows.
    Also latches the node's mode (1 Hz command heartbeat) and feeds the
    planner topic at 10 Hz while in planner mode (planner_cmd_cb supplies the
    command; default is IDLE stand-in-place)."""

    def __init__(self, motion, link, lookahead, mode_cb=lambda: MODE_POSE,
                 planner_cmd_cb=None):
        self.motion, self.link, self.lookahead = motion, link, lookahead
        self.mode_cb = mode_cb
        self.planner_cmd_cb = planner_cmd_cb or (
            lambda: (0, np.zeros(3), np.array([1.0, 0.0, 0.0]), -1.0))
        self.last_planner_cmd = None    # what was last sent, for UI/logging
        self.t0 = None
        self.published = -1
        self.last_heartbeat = -1.0
        self.last_planner = -1.0

    def tick(self, now):
        if self.t0 is None:
            self.t0 = now
        want = int((now - self.t0) * POLICY_FPS) + 1 + self.lookahead
        self.motion.ensure(want)
        newest = self.motion.timesteps - 1
        if newest > self.published:                # never resend an unchanged window:
            first = max(0, newest - self.lookahead)  # the node treats that as a catch-up
            self.link.pose(self.motion, first, newest)
            self.published = newest
        mode = self.mode_cb()
        if mode in (MODE_PLANNER, MODE_POSE):
            if now - self.last_heartbeat >= 1.0:   # keeps the mode latched on the node
                self.link.command(planner=(mode == MODE_PLANNER))
                self.last_heartbeat = now
            if mode == MODE_PLANNER and now - self.last_planner >= 0.1:
                loco, movement, facing, speed = self.planner_cmd_cb()
                self.link.planner_command(loco, movement, facing, speed)
                prev = self.last_planner_cmd
                self.last_planner_cmd = (loco, movement, facing, speed)
                if prev is None or prev[0] != loco:
                    print(f'[bridge] planner cmd -> {LOCO_NAMES.get(loco, loco)} '
                          f'move=({movement[0]:+.2f},{movement[1]:+.2f}) '
                          f'face=({facing[0]:+.2f},{facing[1]:+.2f}) '
                          f'speed={speed:.2f}')
                self.last_planner = now
        else:
            self.last_planner_cmd = None
        return newest


def reroot_matcher(matcher):
    """Move the freshly-reset matcher world so the character starts at the
    origin facing +x, same as the robot spawn in run_sim.py. Call right after
    matcher.reset(); transforms every world-frame field reset() produced."""
    dyaw_quat = yaw_quat(-matcher.rootYaw)
    p0 = matcher.rootPos.copy()

    def xform_point(p):
        q = quat.mul_vec(dyaw_quat, p - p0)
        q[2] = p[2]                    # yaw rotation about z: keep heights
        return q

    matcher.rootPos = xform_point(matcher.rootPos)
    matcher.rootYaw = 0.0
    matcher.rootRot = yaw_quat(0.0)
    matcher.desiredDir = quat.mul_vec(dyaw_quat, matcher.desiredDir)
    matcher.boxPos = xform_point(matcher.boxPos)
    matcher.boxRot = quat.mul(dyaw_quat, matcher.boxRot)
    matcher.Tpos = np.array([xform_point(p) for p in matcher.Tpos])
    matcher.Tdir = np.array([quat.mul_vec(dyaw_quat, v) for v in matcher.Tdir])


def build_matcher(args):
    if args.box_fwd is not None:
        C.BOX_SPAWN_FWD = args.box_fwd
    if args.box_lat is not None:
        C.BOX_SPAWN_LAT = args.box_lat
    print('Loading motion library...')
    lib = load_library()
    matcher = MotionMatcher(lib)
    reroot_matcher(matcher)
    print(f'  {len(lib["qpos"])} frames; box belief {C.BOX_SPAWN_FWD:.2f} m ahead, '
          f'{C.BOX_SPAWN_LAT:+.2f} m lateral of the start pose')
    return matcher


def run_headless(args):
    matcher = build_matcher(args)
    speed = float(args.headless_speed)

    def commander(m):
        fwd = np.array([math.cos(m.rootYaw), math.sin(m.rootYaw), 0.0])
        return fwd * speed, np.zeros(3)

    motion = MMMotion(matcher, commander)
    link = DeployLink(args.bind, args.port)
    st = Streamer(motion, link, args.lookahead)   # heartbeat: POSE mode
    print(f'[headless] publishing on {link.endpoint} for {args.headless:.0f} s, '
          f'lookahead {args.lookahead} frames, speed {speed:.2f} m/s')
    if args.auto_start:
        link.command(start=True)
        print('[headless] sent command start=1 planner=0 (straight to POSE)')
    t_end = time.monotonic() + args.headless
    while time.monotonic() < t_end:
        st.tick(time.monotonic())
        time.sleep(0.004)
    if args.auto_stop:
        link.command(stop=True)
        print('[headless] sent command stop=1')
    print(f'[headless] done: {motion.timesteps} frames generated, '
          f'{link.n_pose} pose messages')
    link.close()
    return 0


# stick-figure color per bridge mode
GHOST_RGBA = {
    MODE_IDLE: np.array([0.6, 0.6, 0.6, 0.6], np.float32),      # gray: no control
    MODE_PLANNER: np.array([0.3, 0.6, 1.0, 0.7], np.float32),   # blue: planner stand
    MODE_POSE: np.array([1.0, 0.75, 0.2, 0.7], np.float32),     # orange: tracking
    MODE_STOPPED: np.array([0.45, 0.45, 0.45, 0.5], np.float32),
}
_EYE3 = np.eye(3).ravel()


def run_window(args):
    import glfw
    import mujoco
    from mm_g1.viewer import InteractiveViewer, draw_gizmos

    matcher = build_matcher(args)
    scene = C.SCENE_BOX_SCENEBOT_XML if C.SCENEBOT_PICK else C.SCENE_BOX_XML
    model = mujoco.MjModel.from_xml_path(scene)
    if C.SCENEBOT_PICK:
        model.geom('box_geom').size[:] = C.BOX_HALF
    data = mujoco.MjData(model)

    class Bridge(InteractiveViewer):
        def __init__(self):
            super().__init__(model, data, matcher,
                             title='G1 hardware bridge -- ] start, P mode, O stop')
            self.ctx.free()                      # base class uses fontscale 150
            self.ctx = mujoco.MjrContext(
                model, mujoco.mjtFontScale.mjFONTSCALE_100)
            self.motion = MMMotion(matcher, lambda m: self._command())
            self.link = DeployLink(args.bind, args.port)
            self.robot = RobotState(args.robot_host, args.robot_port)
            self.mode = MODE_IDLE
            self._last_face = np.array([1.0, 0.0, 0.0])
            self.streamer = Streamer(self.motion, self.link, args.lookahead,
                                     mode_cb=lambda: self.mode,
                                     planner_cmd_cb=self._planner_cmd)
            self.anchor = None
            self._wire_reference()
            self.heading_frame = args.frame in ('heading', 'body')
            # actual-robot display: base xy at the origin (no odometry), z set
            # each frame so the lowest foot point touches the floor
            self._foot_geoms = [
                g for g in range(model.ngeom)
                if model.geom_bodyid[g] in
                (model.body('left_ankle_roll_link').id,
                 model.body('right_ankle_roll_link').id)]
            self._corners = np.array(
                [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                 for sz in (-1, 1)], dtype=float)               # (8, 3)
            # stick figure: FK-only copy of the scene, robot bodies only
            self.gdata = mujoco.MjData(model)
            self.ghost_bodies = [
                b for b in range(1, model.nbody)
                if 'box' not in (mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, b) or '')]

        @property
        def in_planner_mode(self):
            return self.mode == MODE_PLANNER

        def _wire_reference(self):
            """Start alignment + the anchor-xy reference mode.

            Alignment: shift the matcher world so the reference pelvis at
            frame 0 sits at the origin, where the robot is displayed.
            anchor-xy: ContinuousAnchor from ref_modes pulls the reference's
            currently-tracked frame toward the displayed robot's xy (the
            origin). Display-only: the v1 pose stream carries no root xy."""
            self.motion.ensure(1)
            off = self.motion.qpos[0, 0:2].copy()
            self.matcher.rootPos[0:2] -= off
            self.matcher.boxPos[0:2] -= off
            self.matcher.Tpos[:, 0:2] -= off
            self.motion.shift_xy(off)
            if self.has_box:
                self.data.qpos[36:43] = self.matcher.box_qpos()
            if args.ref_mode == 'anchor-xy':
                shim = self

                class _Tracked:      # ContinuousAnchor wants policy.current_frame
                    @property
                    def current_frame(_):
                        return max(0, shim.streamer.published - args.lookahead)

                self.anchor = RM.ContinuousAnchor(
                    self.matcher, self.data, self.motion, _Tracked(),
                    gain=args.anchor_gain)
                self.motion.pre_tick = self.anchor.before_matcher_tick

        def _reset_reference(self):
            self.matcher.reset()
            reroot_matcher(self.matcher)
            self.motion = MMMotion(matcher, lambda m: self._command())
            self.streamer = Streamer(self.motion, self.link, args.lookahead,
                                     mode_cb=lambda: self.mode,
                                     planner_cmd_cb=self._planner_cmd)
            self._wire_reference()
            self.cam.lookat[:] = [0.0, 0.0, 0.8]
            print('[bridge] reference reset to the spawn pose (frames restart at 0)')

        def _on_key(self, window, key, scancode, action, mods):
            if action == glfw.PRESS and key == glfw.KEY_SPACE:
                if self.mode == MODE_POSE:
                    print('[bridge] Space (reset) blocked in POSE mode: teleporting '
                          'the tracked reference would make the robot jump. '
                          'Press P (planner mode) first.')
                else:
                    self._reset_reference()
                return
            if action == glfw.PRESS and key == glfw.KEY_RIGHT_BRACKET:
                if self.mode == MODE_STOPPED:
                    print('[bridge] stopped (damping); restart deploy.sh to resume')
                    return
                self.link.command(start=True, planner=True)
                self.mode = MODE_PLANNER
                print('[bridge] command start=1 planner=1 -> PLANNER (stand)')
                return
            if action == glfw.PRESS and key == glfw.KEY_P:
                if self.mode == MODE_PLANNER:
                    self.link.command(planner=False)
                    self.mode = MODE_POSE
                    print('[bridge] command planner=0 -> POSE (stream tracks)')
                elif self.mode == MODE_POSE:
                    self.link.command(planner=True)
                    self.mode = MODE_PLANNER
                    print('[bridge] command planner=1 -> PLANNER (stand)')
                else:
                    print('[bridge] P ignored: press ] first')
                return
            if action == glfw.PRESS and key == glfw.KEY_O:
                self.link.command(stop=True, planner=self.in_planner_mode)
                self.mode = MODE_STOPPED
                print('[bridge] command stop=1 -> damping')
                return
            if action == glfw.PRESS and key == glfw.KEY_ENTER:
                self.link.command(planner=self.in_planner_mode)
                print(f'[bridge] mode command re-sent (planner='
                      f'{int(self.in_planner_mode)})')
                return
            if action == glfw.PRESS and key == glfw.KEY_F:
                self.heading_frame = not self.heading_frame
                print(f'[bridge] WASD frame -> '
                      f'{"heading" if self.heading_frame else "camera"}')
                return
            super()._on_key(window, key, scancode, action, mods)

        def _command(self):
            if not self.heading_frame:
                vel, face = super()._command()
            else:
                saved = self.cam.azimuth
                self.cam.azimuth = math.degrees(self.matcher.rootYaw)
                vel, face = super()._command()
                self.cam.azimuth = saved
            self._speed = float(np.linalg.norm(vel))
            return vel, face

        def _planner_cmd(self):
            """Planner-mode locomotion from the held keys (like the upstream
            Pico joystick): WASD -> movement + speed, arrows -> facing."""
            vel, face = self._command()
            speed = float(np.linalg.norm(vel))
            if speed < 0.05:
                loco, movement, sp = 0, np.zeros(3), -1.0   # IDLE
            else:
                movement = vel / speed
                loco = 1 if speed < 0.8 else (2 if speed < 2.5 else 3)
                sp = speed
            f = float(np.linalg.norm(face))
            if f > 1e-6:
                self._last_face = face / f
            elif speed >= 0.05:
                self._last_face = movement.copy()
            return loco, movement, self._last_face, sp

        def _pose_robot(self):
            """Main model shows the actual robot (grounded at the origin, same
            look as run_sim.py); default standing pose when no data."""
            state = self.robot.get()
            self.data.qpos[0:2] = 0.0
            self.data.qpos[2] = 1.0                 # placeholder, grounded below
            if state is None:
                self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
                self.data.qpos[7:36] = DEFAULT_QJ
                live = False
            else:
                bq, q = state
                self.data.qpos[3:7] = bq
                self.data.qpos[7:36] = q
                live = True
            mujoco.mj_kinematics(self.model, self.data)
            foot_low = np.inf
            for g in self._foot_geoms:
                c = self.model.geom_aabb[g, :3]     # geom-frame center
                h = self.model.geom_aabb[g, 3:]     # geom-frame half-size
                R = self.data.geom_xmat[g].reshape(3, 3)
                pts = self.data.geom_xpos[g] + (c + self._corners * h) @ R.T
                foot_low = min(foot_low, float(pts[:, 2].min()))
            self.data.qpos[2] -= foot_low - 0.002
            return live

        def _draw_stick_figure(self, cur):
            gq = self.motion.qpos[cur]                # (36+,) kinematic target
            g = self.gdata
            g.qpos[0:36] = gq[0:36]
            mujoco.mj_kinematics(self.model, g)
            for b in self.ghost_bodies:
                pa = self.model.body_parentid[b]
                if pa == 0 or self.scene.ngeom >= self.scene.maxgeom:
                    continue
                a, c = g.xpos[pa], g.xpos[b]
                if np.linalg.norm(c - a) < 1e-6:
                    continue
                gm = self.scene.geoms[self.scene.ngeom]
                mujoco.mjv_initGeom(gm, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                    np.zeros(3), np.zeros(3), _EYE3,
                                    GHOST_RGBA[self.mode])
                mujoco.mjv_connector(gm, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.025,
                                     np.asarray(a, float), np.asarray(c, float))
                self.scene.ngeom += 1

        KEYS_LEFT = (']\nP\nO\nSpace\nEnter\nW A S D\nArrows\nShift\nF\nB / N\nT\nEsc')
        KEYS_RIGHT = ('start control (planner mode)\n'
                      'toggle planner <-> POSE\n'
                      'stop -> damping (final)\n'
                      'reset reference (blocked in POSE)\n'
                      're-send mode command\n'
                      'move\n'
                      'face direction\n'
                      'hold to walk\n'
                      'toggle WASD frame (heading/camera)\n'
                      'box pick: walk-over / instant\n'
                      'toggle gizmos\n'
                      'quit')

        def _status(self, first, newest):
            mode_line = {
                MODE_IDLE: 'IDLE            next: press ] to start (planner mode)',
                MODE_PLANNER: 'PLANNER         robot standing; next: P -> POSE',
                MODE_POSE: 'POSE            motion matching drives the robot',
                MODE_STOPPED: 'STOPPED         damping; restart deploy.sh to resume',
            }[self.mode]
            s = self.matcher.state
            head = s.name if s is not State.LOCOMOTION else (
                'RUN' if self._speed > C.MAX_SPEED * (1 + C.WALK_SCALE) / 2
                else ('WALK' if self._speed > 1e-3 else 'IDLE'))
            ref_line = (f'{head}  {self._speed:.1f} m/s   frames {first}..{newest}'
                        f'   (robot lags {args.lookahead / POLICY_FPS:.1f} s)'
                        f'   ref-mode {args.ref_mode}   wasd-frame '
                        f'{"heading" if self.heading_frame else "camera"}')
            labels = 'mode\nreference\nlink'
            values = f'{mode_line}\n{ref_line}\n{self.link.endpoint}'
            if self.mode == MODE_PLANNER and self.streamer.last_planner_cmd:
                loco, mv, fc, sp = self.streamer.last_planner_cmd
                labels += '\nplanner cmd'
                values += (f'\n{LOCO_NAMES.get(loco, loco)}'
                           f'  move ({mv[0]:+.2f},{mv[1]:+.2f})'
                           f'  face ({fc[0]:+.2f},{fc[1]:+.2f})'
                           f'  speed {sp:.2f} m/s')
            return labels, values

        def run(self, max_frames=None):
            rendered = 0
            while not glfw.window_should_close(self.window):
                if max_frames is not None and rendered >= max_frames:
                    break
                rendered += 1
                newest = self.streamer.tick(glfw.get_time())
                cur = max(0, newest - args.lookahead)   # wall-clock target frame

                live = self._pose_robot()
                q = self.motion.qpos[cur]
                if self.has_box:
                    self.data.qpos[36:43] = q[36:43]
                if self._box_gid is not None:
                    on = (self.matcher.lib['contact'][self.matcher.cur][2:4] > 0.5).any()
                    self.model.geom_rgba[self._box_gid] = (
                        np.array([0.25, 0.9, 0.35, 1.0]) if on else self._box_rgba)
                mujoco.mj_forward(self.model, self.data)
                self.cam.lookat[0] = float(q[0])        # follow the stick figure
                self.cam.lookat[1] = float(q[1])

                w, h = glfw.get_framebuffer_size(self.window)
                viewport = mujoco.MjrRect(0, 0, w, h)
                mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                       mujoco.mjtCatBit.mjCAT_ALL, self.scene)
                self._draw_stick_figure(cur)
                if self.show_traj:
                    # gizmos are drawn at the matcher's CURRENT root, which
                    # runs lookahead ahead of the displayed frame (and gets
                    # anchor corrections): shift them onto the stick figure
                    n0 = self.scene.ngeom
                    draw_gizmos(self.scene, self.matcher)
                    dxy = q[0:2] - self.matcher.rootPos[0:2]  # (2,)
                    for i in range(n0, self.scene.ngeom):
                        self.scene.geoms[i].pos[0] += dxy[0]
                        self.scene.geoms[i].pos[1] += dxy[1]
                mujoco.mjr_render(viewport, self.scene, self.ctx)
                first = max(0, newest - args.lookahead)
                labels, values = self._status(first, newest)
                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    viewport, labels, values, self.ctx)
                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    viewport, self.KEYS_LEFT, self.KEYS_RIGHT, self.ctx)
                if live:
                    mujoco.mjr_text(mujoco.mjtFont.mjFONT_SHADOW,
                                    'ROBOT POSE: LIVE', self.ctx,
                                    0.40, 0.955, 0.25, 1.0, 0.4)
                else:
                    mujoco.mjr_text(mujoco.mjtFont.mjFONT_SHADOW,
                                    'ROBOT POSE: NO DATA (default pose)', self.ctx,
                                    0.36, 0.955, 1.0, 0.55, 0.2)
                glfw.swap_buffers(self.window)
                glfw.poll_events()
            self.robot.close()
            self.link.close()
            glfw.terminate()

    Bridge().run(max_frames=args.smoke_frames)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bind', default='*', help='PUB bind address (default: all interfaces)')
    ap.add_argument('--port', type=int, default=5556,
                    help='deploy node --zmq-port (default 5556)')
    ap.add_argument('--robot-host', default='localhost',
                    help='deploy node g1_debug host (default localhost)')
    ap.add_argument('--robot-port', type=int, default=5557,
                    help='deploy node g1_debug port (default 5557)')
    ap.add_argument('--lookahead', type=int, default=50,
                    help='frames the matcher runs ahead of the clock (50 = 1 s; the '
                         'SONIC encoder looks 45 frames ahead)')
    ap.add_argument('--frame', choices=('heading', 'camera', 'body'),
                    default='heading',
                    help='WASD frame: heading = relative to the reference\'s '
                         'current heading (default; "body" is a legacy alias), '
                         'camera = relative to the view. Key F toggles at runtime')
    ap.add_argument('--ref-mode', choices=('anchor-xy', 'none'), default='none',
                    help='anchor-xy: pull the reference xy toward the displayed '
                         'robot every matcher tick (ContinuousAnchor; display '
                         'only, the pose stream carries no root xy)')
    ap.add_argument('--anchor-gain', type=float, default=0.20,
                    help='anchor-xy correction gain per matcher tick')
    ap.add_argument('--box-fwd', type=float, default=None,
                    help='box belief: metres ahead of the start pose (config default '
                         f'{C.BOX_SPAWN_FWD})')
    ap.add_argument('--box-lat', type=float, default=None,
                    help='box belief: metres to the left (+) of the start pose')
    ap.add_argument('--headless', type=float, default=None, metavar='SECONDS',
                    help='no window: stream a scripted reference for SECONDS and exit')
    ap.add_argument('--headless-speed', type=float, default=0.0,
                    help='forward speed of the headless reference (m/s, default idle)')
    ap.add_argument('--auto-start', action='store_true',
                    help='headless: send start=1 at launch (moves the robot!)')
    ap.add_argument('--auto-stop', action='store_true',
                    help='headless: send stop=1 before exiting')
    ap.add_argument('--smoke-frames', type=int, default=None,
                    help='(testing) close the window after N rendered frames')
    args = ap.parse_args()
    if args.lookahead < 1:
        ap.error('--lookahead must be >= 1')
    if args.headless is not None:
        return run_headless(args)
    return run_window(args)


if __name__ == '__main__':
    sys.exit(main())
