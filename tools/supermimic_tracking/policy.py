"""SuperMimic actor inference through the pinned browser demo runtime.

Reference inputs follow the existing SONIC RefMotion interface. Robot state
is read from native MuJoCo. Dynamics always run in MuJoCo, not the world model.
"""
import atexit
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np

from sonic_tracking import params as P
from sonic_tracking.policy import SonicPolicy
from sonic_tracking.rotations import quat_mul

ASSETS = Path(os.environ.get('SUPERMIMIC_ASSETS',
    Path(__file__).resolve().parents[2] / 'assets/supermimic/demo'))


class SuperMimicPolicy(SonicPolicy):
    """Reuse heading/playback conventions, with SuperMimic's own inference.

    Requires bind_mujoco before step because SuperMimic uses full root state.
    Its native gains and torque limits are exposed in MuJoCo joint order.
    """
    def __init__(self, assets=None):
        self.assets = Path(assets or ASSETS)
        node = shutil.which('node')
        if node is None:
            raise RuntimeError('SuperMimic requires Node.js 18 or newer')
        scene = self.assets / 'scene.xml'
        if not scene.exists():
            raise FileNotFoundError(f'Missing {scene}. Run tools/supermimic_tracking/fetch.py')
        if hashlib.sha256(scene.read_bytes()).hexdigest() != '52c5c169a6516334144db449a3c220349cc6435bef1a260bf46652e0f1474cb0':
            raise RuntimeError('Unsupported SuperMimic scene version')
        self.ref_model = mujoco.MjModel.from_xml_path(str(scene))
        self.ref_data = mujoco.MjData(self.ref_model)
        names = [self.ref_model.joint(i).name for i in range(1, self.ref_model.njnt)]
        if names != list(P.JOINT_NAMES_MUJOCO):
            raise RuntimeError('SuperMimic joint order differs from the project')
        controls = [self.ref_model.actuator(n).id for n in names]
        self.kps = self.ref_model.actuator_gainprm[controls, 0].copy()
        self.kds = -self.ref_model.actuator_biasprm[controls, 2].copy()
        self.torque_limits = self.ref_model.actuator_forcerange[controls].copy()
        constants = json.loads((self.assets / 'supermimic-constants.json').read_text())
        self.body_ids = [self.ref_model.body(n).id for n in constants['bodies']['asset']]
        if len(self.body_ids) != 30 or constants['control_dt'] != P.CONTROL_DT:
            raise RuntimeError('Unsupported body order or control frequency')
        self.process = subprocess.Popen(
            [node, str(Path(__file__).with_name('runtime.mjs')), str(self.assets)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        if self.process.stdout.readline() != b'READY\n':
            self.close()
            raise RuntimeError('SuperMimic runtime initialization failed (see stderr)')
        atexit.register(self.close)
        self.streaming = True
        self.hist = []
        self.last_action = np.zeros(29)
        self.heading_init_base_quat = np.array([1., 0., 0., 0.])
        self.init_ref_root_quat = self.heading_init_base_quat.copy()
        self.delta_heading = 0.
        self._reset_pending = True

    def bind_mujoco(self, model, data):
        self.data = data
        self.q_at = np.array([model.joint(n).qposadr[0] for n in P.JOINT_NAMES_MUJOCO])
        self.dq_at = np.array([model.joint(n).dofadr[0] for n in P.JOINT_NAMES_MUJOCO])
        # The existing scenes use torque motors, in the same named joint order.
        names = [model.joint(int(model.actuator_trnid[i, 0])).name for i in range(29)]
        if names != list(P.JOINT_NAMES_MUJOCO):
            raise RuntimeError('Physical actuator order mismatch')

    def set_motion(self, motion):
        super().set_motion(motion)
        self._reset_pending = True

    def step(self, base_quat, base_ang_vel, joint_pos, joint_vel):
        if not hasattr(self, 'data'):
            raise RuntimeError('Call bind_mujoco(model, data) before SuperMimic inference')
        if self.motion.timesteps < 1:
            raise RuntimeError('Reference stream has no frames')
        reset = self._reset_pending or self.reinitialize_heading
        self._update_heading(base_quat)
        adh = self._apply_delta_heading()
        f = min(self.current_frame, self.motion.timesteps - 1)
        idx = np.minimum(f + (np.arange(46) if self.play else 0), self.motion.timesteps - 1)
        idx = np.broadcast_to(idx, (46,))
        jp = self.motion.joint_pos[idx][:, P.ISAACLAB_TO_MUJOCO]
        jv = self.motion.joint_vel[idx][:, P.ISAACLAB_TO_MUJOCO] if self.play else np.zeros_like(jp)
        bp = np.empty((46, 30, 3))
        bq = np.empty((46, 30, 4))
        # FK uses the demo G1's named bodies. Cached identical paused frames
        # avoid 46 identical FK calls during settling and end-of-clip hold.
        previous = None
        for k, source_frame in enumerate(idx):
            if source_frame == previous:
                bp[k], bq[k] = bp[k-1], bq[k-1]
                continue
            self.ref_data.qpos[:3] = self.motion.body_pos[source_frame, 0]
            self.ref_data.qpos[3:7] = quat_mul(adh, self.motion.body_quat[source_frame, 0])
            self.ref_data.qpos[7:] = jp[k]
            mujoco.mj_kinematics(self.ref_model, self.ref_data)
            bp[k] = self.ref_data.xpos[self.body_ids]
            bq[k] = self.ref_data.xquat[self.body_ids]
            previous = source_frame
        state = np.r_[float(reset), self.data.qpos[:3], base_quat, joint_pos,
                      self.data.qvel[:3], base_ang_vel, joint_vel].astype('<f8')
        refs = np.concatenate([x.ravel() for x in (jp, jv, bp, bq)]).astype('<f4')
        payload = state.tobytes() + refs.tobytes()
        try:
            # Unbuffered pipes can make partial writes/reads.
            view = memoryview(payload)
            while view:
                n = self.process.stdin.write(view)
                if not n:
                    raise BrokenPipeError('Runtime stopped receiving state')
                view = view[n:]
            reply = bytearray()
            while len(reply) < 29*8:
                block = self.process.stdout.read(29*8-len(reply))
                if not block:
                    raise BrokenPipeError('Runtime stopped producing targets')
                reply.extend(block)
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError('SuperMimic inference failed (see stderr)') from exc
        target = np.frombuffer(reply, dtype='<f8').copy()
        if not np.isfinite(target).all():
            raise RuntimeError('Non-finite SuperMimic output')
        self._reset_pending = False
        if self.play:
            self.current_frame = min(self.current_frame + 1, self.motion.timesteps - 1)
        return target

    def clip_torque(self, torque):
        return np.clip(torque, self.torque_limits[:, 0], self.torque_limits[:, 1])

    def close(self):
        process = getattr(self, 'process', None)
        if process is not None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=3)
            if process.stdout:
                process.stdout.close()
            self.process = None
        atexit.unregister(self.close)
