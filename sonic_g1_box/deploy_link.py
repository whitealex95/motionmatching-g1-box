"""The deploy node's ZMQ wire (a mirror of motionmatching-g1-deploy/tools/zmq_protocol.py) and the desktop's JSON bus
(a mirror of motionmatching-g1-deploy/ego_slam/wire.py)."""
import json
import threading
import time

import msgpack
import numpy as np
import zmq

HEADER_SIZE = 1280
WIRE_VERSION = 1
_DTYPE_STR = {'float32': 'f32', 'float64': 'f64', 'int32': 'i32', 'int64': 'i64', 'uint8': 'u8', 'bool': 'bool'}

MODE_IDLE = 'idle'          # the node waits in WAIT_FOR_CONTROL
MODE_PLANNER = 'planner'    # the node's planner mode, the robot stands
MODE_POSE = 'pose'          # STREAMED_MOTION: the node tracks the pose stream
MODE_STOPPED = 'stopped'    # damping sent, deploy.sh must be restarted

ROBOT_STALE_S = 0.5
BUS_PUB = 'tcp://127.0.0.1:5581'


def pack_message(topic, arrays):
    fields, blobs = [], []
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        if arr.dtype.byteorder == '>':
            arr = arr.astype(arr.dtype.newbyteorder('<'))
        fields.append({'name': name, 'dtype': _DTYPE_STR[arr.dtype.name], 'shape': list(arr.shape)})
        blobs.append(arr.tobytes())
    header = json.dumps({'v': WIRE_VERSION, 'endian': 'le', 'count': 1, 'fields': fields}).encode()
    if len(header) > HEADER_SIZE:
        raise ValueError(f'header too large: {len(header)} > {HEADER_SIZE}')
    return topic.encode() + header.ljust(HEADER_SIZE, b'\0') + b''.join(blobs)


class BusPublisher:
    """Topic = the name without its leading '/', '/' -> '.'. Never blocks: a gone bus drops the message."""

    def __init__(self, endpoint=BUS_PUB):
        self.sock = zmq.Context.instance().socket(zmq.PUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(endpoint)
        time.sleep(0.05)

    def publish(self, name, payload):
        raw = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        try:
            self.sock.send_multipart([name.strip('/').replace('/', '.').encode('utf-8'), raw], flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self):
        self.sock.close(0)


class DeployLink:
    """PUB side of the node's zmq_manager: the command, planner, pose and arm_tau topics."""

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
        self.sock.send(pack_message('command', {'start': u8(start), 'stop': u8(stop), 'planner': u8(planner)}))

    def planner_command(self, mode, movement, facing, speed=-1.0):
        """mode: the node's LocomotionMode (0 IDLE, 1 SLOW_WALK, 2 WALK, 3 RUN)."""
        self.sock.send(pack_message('planner', {
            'mode': np.array([mode], dtype=np.int32),
            'movement': np.asarray(movement, dtype=np.float32),       # (3,) unit direction
            'facing': np.asarray(facing, dtype=np.float32),           # (3,) unit direction
            'speed': np.array([speed], dtype=np.float32),
            'height': np.array([-1.0], dtype=np.float32),
        }))

    def arm_tau(self, tau):
        """tau: (14,) N*m on arm joints 15..28 in MuJoCo order. The node adds tau / kp to SONIC's arm targets."""
        self.sock.send(pack_message('arm_tau', {'tau': np.asarray(tau, dtype=np.float32)}))

    def pose(self, joint_pos, joint_vel, body_quat, frame_index):
        self.sock.send(pack_message('pose', {
            'joint_pos': np.asarray(joint_pos, dtype=np.float32),       # (N, 29) IsaacLab order
            'joint_vel': np.asarray(joint_vel, dtype=np.float32),       # (N, 29)
            'body_quat': np.asarray(body_quat, dtype=np.float32),       # (N, 4) w, x, y, z
            'frame_index': np.asarray(frame_index, dtype=np.int64),     # (N,)
        }))
        self.n_pose += 1

    def close(self):
        self.sock.close(linger=200)
        self.ctx.term()


class RobotState:
    """SUB side of the node's g1_debug output (port 5557, msgpack)."""

    def __init__(self, host='localhost', port=5557, topic='g1_debug'):
        self.topic = topic.encode()
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, 10)
        self.sock.setsockopt(zmq.SUBSCRIBE, self.topic)
        self.sock.connect(f'tcp://{host}:{port}')
        self.lock = threading.Lock()
        self.base_quat = None        # (4,) w, x, y, z
        self.body_q = None           # (29,) MuJoCo order
        self.heading = None          # (init_base_quat (4,), delta_heading): the node's heading state, once it has one
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
                d = msgpack.unpackb(msg[len(self.topic):], raw=False)
                quat = np.asarray(d['base_quat'], dtype=np.float64)      # (4,)
                q = np.asarray(d['body_q'], dtype=np.float64)            # (29,)
                heading = ((np.asarray(d['init_base_quat'], dtype=np.float64), float(d.get('delta_heading', 0.0)))
                           if 'init_base_quat' in d else None)
            except Exception:
                continue
            with self.lock:
                self.base_quat, self.body_q, self.heading = quat, q, heading
                self.stamp = time.monotonic()

    def get(self):
        """(base_quat (4,), body_q (29,)), or None when there is none or it is older than ROBOT_STALE_S."""
        with self.lock:
            if self.body_q is None or time.monotonic() - self.stamp > ROBOT_STALE_S:
                return None
            return self.base_quat.copy(), self.body_q.copy()

    def heading_state(self):
        """(base_quat (4,), init_base_quat (4,), delta_heading), or None without a fresh one."""
        with self.lock:
            if self.heading is None or time.monotonic() - self.stamp > ROBOT_STALE_S:
                return None
            return self.base_quat.copy(), self.heading[0].copy(), self.heading[1]

    def close(self):
        self._stop = True
        self.thread.join(timeout=1.0)
        self.sock.close(linger=0)
        self.ctx.term()
