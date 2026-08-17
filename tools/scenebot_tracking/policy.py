"""ScenebotPolicy -- the SceneBot demo's ONNX tracking policy (obs 160 ->
actions 29), ported from the web demo's JS wrapper.

Per 50 Hz tick: ingest the stream packet from the motion-graph player, read
the robot state from MuJoCo, assemble the observation in meta.obs_names
order, run the MLP, and map the Isaac-order action to a MuJoCo-order PD
target (torque PD at 200 Hz is done by the caller with meta joint gains).
"""
import numpy as np
import onnxruntime as ort

from . import params as P
from .rotations import (quat_conj_xyzw, quat_rotate_xyzw, quat_mul_xyzw,
                        quat_to_mat33_xyzw)


def _expand_4way_to_8(c):
    out = np.zeros(8, np.float32)
    out[0], out[2], out[5], out[7] = c[0], c[1], c[2], c[3]
    return out


def expand_contact(label, meta=P.META):
    """The demo's UY(): normalize a raw contact label to meta.contact_dim."""
    dim = int(meta['contact_dim'])
    c = np.asarray(label, np.float32)
    use10 = bool(meta.get('use_10way_contact'))
    use5from4 = bool(meta.get('use_5dim_contact_from_4dim'))
    use8 = bool(meta.get('use_8way_contact'))
    if use10 and use5from4:
        if len(c) == 4:
            c = np.concatenate([_expand_4way_to_8(c), np.zeros(2, np.float32)])
        elif len(c) == 5:
            c = np.concatenate([_expand_4way_to_8(c[:4]),
                                np.zeros(2, np.float32)])
        elif len(c) == 8:
            c = np.concatenate([c, np.zeros(2, np.float32)])
        elif len(c) == 10:
            c = np.concatenate([c[:8], np.zeros(2, np.float32)])
        else:
            raise ValueError(f'contact label of {len(c)} values')
    elif use10:
        if len(c) == 5:
            out = np.zeros(10, np.float32)
            out[:8] = _expand_4way_to_8(c[:4])
            out[8] = c[4]
            c = out
        if len(c) != dim:
            raise ValueError(f'contact label of {len(c)} values, want {dim}')
    elif use5from4:
        if len(c) == 4:
            c = np.concatenate([c, np.zeros(1, np.float32)])
        elif len(c) == 5:
            c = c.copy()
            c[4] = 0.0
        if len(c) != dim:
            raise ValueError(f'contact label of {len(c)} values, want {dim}')
    elif use8:
        if len(c) == 4:
            c = _expand_4way_to_8(c)
        if len(c) != dim:
            raise ValueError(f'contact label of {len(c)} values, want {dim}')
    elif len(c) != dim:
        raise ValueError(f'contact label of {len(c)} values, want {dim}')
    return c.astype(np.float32)


class ScenebotPolicy:
    def __init__(self, onnx_path=P.POLICY_ONNX, meta=P.META):
        self.meta = meta
        self.session = ort.InferenceSession(
            onnx_path, providers=['CPUExecutionProvider'])
        self._in = self.session.get_inputs()[0].name
        self._out = self.session.get_outputs()[0].name
        self.reset()

    def reset(self):
        m = self.meta
        self.last_action = np.zeros(P.ACTION_DIM, np.float32)
        self.lower_cmd = np.zeros(int(m['lower_cmd_dim']), np.float32)
        self.vr_pos = np.zeros(int(m['vr_pos_dim']), np.float32)
        self.vr_orn = np.zeros(int(m['vr_orn_dim']), np.float32)
        self.vr_orn[0::4] = 1.0          # identity wxyz per VR body
        self.contact_mask = expand_contact(m.get('default_contact_label', []),
                                           m)
        self.anchor_pos_w = np.zeros(3)
        self.anchor_orn_w = np.array([0.0, 0.0, 0.0, 1.0])   # xyzw

    def ingest(self, pkt):
        if 'lower_cmd' in pkt:
            self.lower_cmd = np.asarray(pkt['lower_cmd'], np.float32)
        if 'vr_3point_pos_l' in pkt:
            self.vr_pos = np.asarray(pkt['vr_3point_pos_l'], np.float32)
        if 'vr_3point_orn_l' in pkt:
            self.vr_orn = np.asarray(pkt['vr_3point_orn_l'], np.float32)
        if 'contact_mask' in pkt:
            self.contact_mask = expand_contact(pkt['contact_mask'], self.meta)
        if 'motion_anchor_pos_w' in pkt:
            self.anchor_pos_w = np.asarray(pkt['motion_anchor_pos_w'], float)
        if 'motion_anchor_orn_w' in pkt:
            self.anchor_orn_w = np.asarray(pkt['motion_anchor_orn_w'], float)

    def control_signals(self, state):
        root_inv = quat_conj_xyzw(state['root_orn_xyzw'])
        rel = self.anchor_pos_w - np.asarray(state['root_pos'], float)
        anchor_b = quat_rotate_xyzw(root_inv, rel)
        q_err = quat_mul_xyzw(root_inv, self.anchor_orn_w)
        R = quat_to_mat33_xyzw(q_err)
        ori_6d = np.array([R[0, 0], R[0, 1], R[1, 0], R[1, 1], R[2, 0],
                           R[2, 1]], np.float32)
        gravity = quat_rotate_xyzw(root_inv, np.array([0.0, 0.0, -1.0]))
        return {
            'lower_command': self.lower_cmd,
            'vr_3point_pos': self.vr_pos,
            'vr_3point_ori': self.vr_orn,
            'contact_mask': self.contact_mask,
            'motion_anchor_pos_b': anchor_b.astype(np.float32),
            'motion_anchor_ori_b': ori_6d,
            'projected_gravity': gravity.astype(np.float32),
        }

    def build_obs(self, state, signals):
        parts = []
        for name in P.OBS_NAMES:
            if name == 'q':
                parts.append(np.asarray(state['q'], np.float32)
                             [P.MUJOCO_TO_ISAAC] - P.DEFAULT_Q_ISAAC)
            elif name == 'dq':
                parts.append(np.asarray(state['dq'], np.float32)
                             [P.MUJOCO_TO_ISAAC])
            elif name == 'last_action':
                parts.append(self.last_action)
            elif name == 'root_vel':
                parts.append(np.asarray(state['root_vel'], np.float32))
            elif name == 'omega':
                parts.append(np.asarray(state['omega'], np.float32))
            elif name in signals:
                parts.append(np.asarray(signals[name], np.float32))
            elif name in state:
                parts.append(np.asarray(state[name], np.float32))
            else:
                raise KeyError(f'unknown obs field "{name}"')
        obs = np.concatenate(parts).astype(np.float32)
        if obs.shape[0] != P.OBS_DIM:
            raise ValueError(f'obs is {obs.shape[0]}, want {P.OBS_DIM}')
        return obs

    def act(self, obs):
        out = self.session.run([self._out],
                               {self._in: obs.reshape(1, -1)})[0][0]
        self.last_action = out.astype(np.float32)
        return self.last_action

    def target_mj(self, action):
        # q_target[mj] = action[isaac(mj)] * scale[mj] + default[isaac(mj)]
        idx = P.ISAAC_TO_MUJOCO
        return (np.asarray(action, float)[idx] * P.ACTION_SCALE_MJ
                + P.DEFAULT_Q_ISAAC[idx])

    def step(self, state):
        signals = self.control_signals(state)
        obs = self.build_obs(state, signals)
        action = self.act(obs)
        return self.target_mj(action)
