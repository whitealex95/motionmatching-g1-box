"""Stream the box motion matcher to the REAL G1 through the C++ deploy node.

Plays the role the Pico manager plays in the lab's teleop stack: this process
binds a ZMQ PUB socket and the deploy node (`motionmatching-g1-deploy`,
`./deploy.sh --input-type zmq_manager real ...`) subscribes to it.

  command topic   {start, stop, planner=0}   planner=0 -> node tracks the pose stream
  pose topic      protocol v1: joint_pos / joint_vel [N,29] (IsaacLab order),
                  body_quat [N,4] (w,x,y,z), frame_index [N] int64, at 50 Hz

The matcher runs LOOKAHEAD frames ahead of wall-clock time, so every chunk
carries real future frames for the SONIC encoder (it looks 45 frames ahead).
The robot therefore reacts to a key press about LOOKAHEAD / 50 s later.

Window (same controls as run.py / run_interactive.py, plus the robot keys):

  W / A / S / D    move   (--frame camera: relative to the view, default;
                           --frame body: relative to the reference's heading)
  Arrow keys       face direction
  Shift (hold)     walk instead of run
  B                box action: walk over + pick up / set down
  N                pick up NOW (skip the walk-over)
  T                toggle gizmos
  ]                START: send command start=1 (deploy node WAIT_FOR_CONTROL -> CONTROL)
  O                STOP:  send command stop=1  (deploy node -> damping; not resumable)
  Enter            re-send the mode command (planner=0)
  Esc              quit this window (the deploy node keeps the last frames)

Space (matcher reset) is disabled here: teleporting the reference would make
the robot jump.

    python run_hardware.py                     # bind tcp://*:5556
    python run_hardware.py --frame body --lookahead 50
    python run_hardware.py --headless 10       # no window, idle reference, 10 s (smoke test)
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
os.environ.setdefault('MUJOCO_GL', 'glfw')

import numpy as np
import zmq

from mm_g1 import config as C
from mm_g1.data import load_library
from mm_g1.controller import MotionMatcher
from mm_g1.states import State
from mm_stream import MMMotion, POLICY_FPS

# --- wire format (mirror of motionmatching-g1-deploy/tools/zmq_protocol.py) ---
HEADER_SIZE = 1280
_DTYPE_STR = {'float32': 'f32', 'float64': 'f64', 'int32': 'i32',
              'int64': 'i64', 'uint8': 'u8', 'bool': 'bool'}


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
    """PUB side of the deploy node's zmq_manager (command + pose topics)."""

    def __init__(self, bind='*', port=5556):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 10)
        self.sock.bind(f'tcp://{bind}:{port}')
        self.endpoint = f'tcp://{bind}:{port}'
        self.n_pose = 0
        time.sleep(0.3)                    # let a waiting subscriber connect

    def command(self, start=False, stop=False):
        u8 = lambda b: np.array([1 if b else 0], dtype=np.uint8)
        self.sock.send(pack_message('command', {
            'start': u8(start), 'stop': u8(stop), 'planner': u8(False)}, version=1))

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


class Streamer:
    """Real-time driver: keeps the matcher LOOKAHEAD frames ahead of the clock
    and publishes the window [newest - lookahead, newest] whenever it grows."""

    def __init__(self, motion, link, lookahead):
        self.motion, self.link, self.lookahead = motion, link, lookahead
        self.t0 = None
        self.published = -1
        self.last_heartbeat = -1.0

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
        if now - self.last_heartbeat >= 1.0:       # keeps planner=0 latched on the node
            self.link.command()
            self.last_heartbeat = now
        return newest


def build_matcher(args):
    if args.box_fwd is not None:
        C.BOX_SPAWN_FWD = args.box_fwd
    if args.box_lat is not None:
        C.BOX_SPAWN_LAT = args.box_lat
    print('Loading motion library...')
    lib = load_library()
    matcher = MotionMatcher(lib)
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
    st = Streamer(motion, link, args.lookahead)
    print(f'[headless] publishing on {link.endpoint} for {args.headless:.0f} s, '
          f'lookahead {args.lookahead} frames, speed {speed:.2f} m/s')
    if args.auto_start:
        link.command(start=True)
        print('[headless] sent command start=1')
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
                             title='G1 hardware bridge -- ] start, O stop, WASD move, B box')
            self.motion = MMMotion(matcher, lambda m: self._command())
            self.link = DeployLink(args.bind, args.port)
            self.streamer = Streamer(self.motion, self.link, args.lookahead)
            self.status = 'idle (press ] to start control on the robot)'
            self.body_frame = args.frame == 'body'

        def _on_key(self, window, key, scancode, action, mods):
            if action == glfw.PRESS and key == glfw.KEY_SPACE:
                print('[bridge] Space (matcher reset) is disabled on hardware')
                return
            if action == glfw.PRESS and key == glfw.KEY_RIGHT_BRACKET:
                self.link.command(start=True)
                self.status = 'START sent (robot in CONTROL)'
                print('[bridge] command start=1')
                return
            if action == glfw.PRESS and key == glfw.KEY_O:
                self.link.command(stop=True)
                self.status = 'STOP sent (robot damping; restart deploy.sh to resume)'
                print('[bridge] command stop=1')
                return
            if action == glfw.PRESS and key == glfw.KEY_ENTER:
                self.link.command()
                print('[bridge] command planner=0 re-sent')
                return
            super()._on_key(window, key, scancode, action, mods)

        def _command(self):
            if not self.body_frame:
                vel, face = super()._command()
            else:
                saved = self.cam.azimuth
                self.cam.azimuth = math.degrees(self.matcher.rootYaw)
                vel, face = super()._command()
                self.cam.azimuth = saved
            self._speed = float(np.linalg.norm(vel))
            return vel, face

        def run(self, max_frames=None):
            rendered = 0
            while not glfw.window_should_close(self.window):
                if max_frames is not None and rendered >= max_frames:
                    break
                rendered += 1
                newest = self.streamer.tick(glfw.get_time())
                q = self.motion.qpos[newest]
                self.data.qpos[0:36] = q[0:36]
                if self.has_box:
                    self.data.qpos[36:43] = q[36:43]
                if self._box_gid is not None:
                    on = (self.matcher.lib['contact'][self.matcher.cur][2:4] > 0.5).any()
                    self.model.geom_rgba[self._box_gid] = (
                        np.array([0.25, 0.9, 0.35, 1.0]) if on else self._box_rgba)
                mujoco.mj_forward(self.model, self.data)
                self.cam.lookat[0] = float(self.data.qpos[0])
                self.cam.lookat[1] = float(self.data.qpos[1])

                w, h = glfw.get_framebuffer_size(self.window)
                viewport = mujoco.MjrRect(0, 0, w, h)
                mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                       mujoco.mjtCatBit.mjCAT_ALL, self.scene)
                if self.show_traj:
                    draw_gizmos(self.scene, self.matcher)
                mujoco.mjr_render(viewport, self.scene, self.ctx)
                self._overlay(viewport, self._speed)
                first = max(0, newest - args.lookahead)
                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    viewport,
                    f'deploy link {self.link.endpoint}   frame {first}..{newest}  '
                    f'(robot lags {args.lookahead / POLICY_FPS:.1f} s)',
                    f'{self.status}\n'
                    f'] start | O stop | Enter re-send mode | frame: {args.frame}',
                    self.ctx)
                glfw.swap_buffers(self.window)
                glfw.poll_events()
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
    ap.add_argument('--lookahead', type=int, default=50,
                    help='frames the matcher runs ahead of the clock (50 = 1 s; the '
                         'SONIC encoder looks 45 frames ahead)')
    ap.add_argument('--frame', choices=('camera', 'body'), default='camera',
                    help='WASD reference frame')
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
