"""SonicPolicy -- the released SONIC ONNX encoder/decoder pair driven by a
pure-Python port of the official C++ observation pipeline
(gear_sonic_deploy/src/g1/g1_deploy_onnx_ref.cpp), G1 reference-motion
tracking mode (encode mode 0).

Faithfully replicated semantics (all verified against the C++ source):
  * observation_config.yaml decides the observation list and offsets; the
    encoder mode's required_observations are computed, everything else is
    ZERO-filled (GatherEncoderObservations).
  * Histories come from a ring of per-tick state entries logged BEFORE
    inference; last_action in an entry is the action of the PREVIOUS tick
    (GatherRobotStateToLogger runs before CreatePolicyCommand).
  * History gathers return frames OLDEST-FIRST, zero-padded in the oldest
    slots when fewer entries exist (StateLogger::GetLatest + newest_first
    defaulting to false in the gatherers).
  * body_q history = isaac(q) - default_angles; body_dq raw; gravity_dir =
    conj(base_quat) rotate (0,0,-1); base ang vel in the base frame.
  * Motion future gathers: positions hold the current frame when paused;
    velocities are zeroed when paused; frames clamp at the end
    (GatherMotionJointPositions/VelocitiesMultiFrame).
  * Anchor orientation = 6D of conj(base_quat) * apply_delta_heading *
    ref_root_quat[frame], where apply_delta_heading aligns the motion's
    initial heading with the robot heading captured at (re)initialisation
    (UpdateHeadingState / ComputeApplyDeltaHeading, orientation_mode 0).
  * Action -> target: q_target_mj = default + action[isaac->mj] * scale;
    frame advances once per tick while playing; at the end play stops,
    the frame resets to 0 and the heading re-initialises
    (CreatePolicyCommand / CurrentFrameAdvancement).
"""

import numpy as np
import onnxruntime as ort
import yaml

from . import params as P
from .rotations import (quat_mul, quat_conjugate, quat_rotate,
                        calc_heading_quat, calc_heading_quat_inv,
                        euler_z_to_quat, quat_to_6d)

# dimensions of every observation name that can appear in the released
# configs (from the C++ GetObservationRegistry); token_state is added at
# runtime with the encoder's dimension.
OBS_DIMS = {
    'encoder_mode': 3, 'encoder_mode_4': 4, 'encoder_index': 1,
    'motion_joint_positions': 29, 'motion_joint_velocities': 29,
    'motion_anchor_orientation': 6, 'motion_root_z_position': 1,
    'motion_root_z_position_10frame_step5': 10,
    'motion_root_z_position_10frame_step1': 10,
    'motion_anchor_orientation_10frame_step5': 60,
    'motion_anchor_orientation_10frame_step1': 60,
    'motion_joint_positions_10frame_step5': 290,
    'motion_joint_velocities_10frame_step5': 290,
    'motion_joint_positions_10frame_step1': 290,
    'motion_joint_velocities_10frame_step1': 290,
    'motion_joint_positions_lowerbody_10frame_step5': 120,
    'motion_joint_velocities_lowerbody_10frame_step5': 120,
    'motion_joint_positions_lowerbody_10frame_step1': 120,
    'motion_joint_velocities_lowerbody_10frame_step1': 120,
    'motion_joint_positions_wrists_10frame_step1': 60,
    'motion_joint_velocities_wrists_10frame_step1': 60,
    'motion_anchor_orientation_heading': 6,
    'motion_anchor_orientation_heading_10frame_step5': 60,
    'motion_anchor_orientation_heading_10frame_step1': 60,
    'motion_joint_positions_wrists_4frame_step1': 24,
    'vr_3point_local_target': 9, 'vr_3point_local_orn_target': 12,
    'vr_3point_compliance': 3,
    'smpl_joints_10frame_step1': 720,
    'smpl_joints_4frame_step1': 288,
    'smpl_anchor_orientation_10frame_step1': 60,
    'smpl_anchor_orientation_4frame_step1': 24,
    'smpl_anchor_orientation_heading_10frame_step1': 60,
    'base_angular_velocity': 3, 'body_joint_positions': 29,
    'body_joint_velocities': 29, 'last_actions': 29, 'gravity_dir': 3,
    'his_base_angular_velocity_10frame_step1': 30,
    'his_body_joint_positions_10frame_step1': 290,
    'his_body_joint_velocities_10frame_step1': 290,
    'his_last_actions_10frame_step1': 290,
    'his_gravity_dir_10frame_step1': 30,
    'his_base_angular_velocity_4frame_step1': 12,
    'his_body_joint_positions_4frame_step1': 116,
    'his_body_joint_velocities_4frame_step1': 116,
    'his_last_actions_4frame_step1': 116,
    'his_gravity_dir_4frame_step1': 12,
}

_HIS_LEN = 10        # all released configs use 10-frame step-1 histories


class _Entry:
    """One per-tick state snapshot (StateLogger::Entry subset)."""
    __slots__ = ('base_quat', 'base_ang_vel', 'body_q', 'body_dq', 'last_action')

    def __init__(self, base_quat=None, base_ang_vel=None, body_q=None,
                 body_dq=None, last_action=None):
        z29 = np.zeros(29)
        # NOTE: the C++ zero entry has an all-zero quaternion (not identity);
        # gravity_dir computed from it is (0,0,0), matching their padding.
        self.base_quat = np.zeros(4) if base_quat is None else base_quat
        self.base_ang_vel = np.zeros(3) if base_ang_vel is None else base_ang_vel
        self.body_q = z29 if body_q is None else body_q
        self.body_dq = z29.copy() if body_dq is None else body_dq
        self.last_action = z29.copy() if last_action is None else last_action


class SonicPolicy:
    def __init__(self, variant=None, policy_dir=None, device='cpu'):
        """variant: one of params.SONIC_VARIANTS ('release', 'low_latency',
        'sonic_v1_1'); policy_dir overrides it with an explicit directory."""
        self.variant = variant or P.DEFAULT_VARIANT
        policy_dir = policy_dir or P.variant_dir(self.variant)
        import os
        enc_path = os.path.join(policy_dir, 'model_encoder.onnx')
        dec_path = os.path.join(policy_dir, 'model_decoder.onnx')
        cfg_path = os.path.join(policy_dir, 'observation_config.yaml')

        if device == 'cuda':
            # onnxruntime's CUDA provider needs libcublasLt / libcudnn on the
            # loader path.  torch ships them under site-packages/nvidia/ and
            # preloads them process-globally on import, so importing it first
            # is what makes the provider resolve.  Without this ort logs
            # "Failed to create CUDAExecutionProvider" and SILENTLY falls back
            # to CPU -- the session still works, just slower, which is easy to
            # miss.  Optional import: the mm-g1-sonic env used by
            # run_tracking.py has no torch and runs the policy on CPU.
            try:
                import torch                                   # noqa: F401
            except ImportError:
                pass
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if device == 'cuda' else ['CPUExecutionProvider'])
        self.enc = ort.InferenceSession(enc_path, providers=providers)
        self.dec = ort.InferenceSession(dec_path, providers=providers)
        if device == 'cuda' and 'CUDAExecutionProvider' not in self.enc.get_providers():
            print('[sonic] WARNING: --device cuda requested but onnxruntime '
                  'fell back to CPU (CUDA provider unavailable); the policy '
                  'will run on CPU')
        self.enc_in = self.enc.get_inputs()[0].name
        self.dec_in = self.dec.get_inputs()[0].name
        enc_dim = int(self.enc.get_inputs()[0].shape[1])
        dec_dim = int(self.dec.get_inputs()[0].shape[1])

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.token_dim = int(cfg['encoder']['dimension'])
        dims = dict(OBS_DIMS, token_state=self.token_dim)

        def layout(entries):
            out, off = [], 0
            for e in entries:
                if not e.get('enabled', False):
                    continue
                name = e['name']
                if name not in dims:
                    raise KeyError(f'unknown observation {name!r}')
                out.append((name, off, dims[name]))
                off += dims[name]
            return out, off

        self.policy_obs, n = layout(cfg['observations'])
        assert n == dec_dim, f'policy obs {n} != decoder input {dec_dim}'
        self.encoder_obs, n = layout(cfg['encoder']['encoder_observations'])
        assert n == enc_dim, f'encoder obs {n} != encoder input {enc_dim}'
        self.enc_dim, self.dec_dim = enc_dim, dec_dim

        self.encode_mode = 0                      # G1 joint tracking
        modes = {m['mode_id']: m for m in cfg['encoder']['encoder_modes']}
        self.required = set(modes[self.encode_mode]['required_observations'])

        # runtime state
        self.motion = None
        self.current_frame = 0
        self.play = False
        self.streaming = False    # streamed motions hold the newest frame
                                  # instead of resetting at the end (the C++
                                  # "streamed" motion special case)
        self.last_action = np.zeros(29)
        self.token_state = np.zeros(self.token_dim)
        self.hist = []                            # ring of _Entry, newest last
        self.reinitialize_heading = True
        self.heading_init_base_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.delta_heading = 0.0
        self.init_ref_root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # -- motion control (keyboard T/R equivalents) ----------------------------
    def set_motion(self, motion):
        self.motion = motion
        self.current_frame = 0
        self.play = False
        self.reinitialize_heading = True

    def start_play(self):
        self.play = True

    # -- heading state (UpdateHeadingState / ComputeApplyDeltaHeading) --------
    def _update_heading(self, base_quat):
        if self.reinitialize_heading:
            self.heading_init_base_quat = base_quat.copy()
            self.delta_heading = 0.0
            self.init_ref_root_quat = self.motion.body_quat[self.current_frame, 0].copy()
            self.reinitialize_heading = False
        if self.current_frame == 0:
            self.init_ref_root_quat = self.motion.body_quat[0, 0].copy()

    def _apply_delta_heading(self):
        adh = quat_mul(calc_heading_quat(self.heading_init_base_quat),
                       calc_heading_quat_inv(self.init_ref_root_quat))
        if self.delta_heading != 0.0:
            adh = quat_mul(euler_z_to_quat(self.delta_heading), adh)
        return adh

    # -- motion future gathers -------------------------------------------------
    def _future_frames(self, num, step, advance_when_paused=False):
        T = self.motion.timesteps
        base = self.current_frame
        idx = []
        for i in range(num):
            f = base + i * step if (self.play or advance_when_paused) else base
            idx.append(min(f, T - 1))
        return np.array(idx)

    def _gather_motion_jpos(self, num, step):
        return self.motion.joint_pos[self._future_frames(num, step)].reshape(-1)

    def _gather_motion_jvel(self, num, step):
        # frames always advance; values zeroed when paused (C++ semantics)
        idx = self._future_frames(num, step, advance_when_paused=True)
        if not self.play:
            return np.zeros(num * 29)
        return self.motion.joint_vel[idx].reshape(-1)

    def _gather_anchor_ori(self, num, step, base_quat, heading_only=False):
        # heading_only is the C++ orientation_mode 1 (motion_anchor_ori_heading,
        # SONIC v1.1): the left quaternion is the robot's heading yaw only.
        adh = self._apply_delta_heading()
        left = calc_heading_quat(base_quat) if heading_only else base_quat
        base_conj = quat_conjugate(left)
        out = np.empty(num * 6)
        for k, f in enumerate(self._future_frames(num, step)):
            new_ref = quat_mul(adh, self.motion.body_quat[f, 0])
            out[k * 6:(k + 1) * 6] = quat_to_6d(quat_mul(base_conj, new_ref))
        return out

    def _gather_root_z(self, num, step):
        return self.motion.body_pos[self._future_frames(num, step), 0, 2]

    # -- history gathers --------------------------------------------------------
    def _hist_frames(self, n=_HIS_LEN):
        """Oldest-first list of the last n entries, zero-padded at the front."""
        got = self.hist[-n:]
        pad = [_Entry() for _ in range(n - len(got))]
        return pad + got

    def _gather_his(self, field, n=_HIS_LEN):
        return np.concatenate([getattr(e, field) for e in self._hist_frames(n)])

    def _gather_gravity_dir(self, n=_HIS_LEN):
        out = np.empty(n * 3)
        for k, e in enumerate(self._hist_frames(n)):
            out[k * 3:(k + 1) * 3] = quat_rotate(quat_conjugate(e.base_quat),
                                                 np.array([0.0, 0.0, -1.0]))
        return out

    # -- observation assembly ----------------------------------------------------
    def _encoder_observations(self, base_quat):
        buf = np.zeros(self.enc_dim)
        for name, off, dim in self.encoder_obs:
            if name not in self.required:
                continue                          # zero-filled for this mode
            if name in ('encoder_mode', 'encoder_mode_4', 'encoder_index'):
                buf[off] = float(self.encode_mode)
            elif name == 'motion_joint_positions_10frame_step5':
                buf[off:off + dim] = self._gather_motion_jpos(10, 5)
            elif name == 'motion_joint_velocities_10frame_step5':
                buf[off:off + dim] = self._gather_motion_jvel(10, 5)
            elif name == 'motion_anchor_orientation_10frame_step5':
                buf[off:off + dim] = self._gather_anchor_ori(10, 5, base_quat)
            elif name == 'motion_anchor_orientation_10frame_step1':
                buf[off:off + dim] = self._gather_anchor_ori(10, 1, base_quat)
            elif name == 'motion_anchor_orientation':
                buf[off:off + dim] = self._gather_anchor_ori(1, 1, base_quat)
            elif name == 'motion_anchor_orientation_heading_10frame_step5':
                buf[off:off + dim] = self._gather_anchor_ori(
                    10, 5, base_quat, heading_only=True)
            elif name == 'motion_anchor_orientation_heading_10frame_step1':
                buf[off:off + dim] = self._gather_anchor_ori(
                    10, 1, base_quat, heading_only=True)
            elif name == 'motion_anchor_orientation_heading':
                buf[off:off + dim] = self._gather_anchor_ori(
                    1, 1, base_quat, heading_only=True)
            elif name == 'motion_joint_positions_10frame_step1':
                buf[off:off + dim] = self._gather_motion_jpos(10, 1)
            elif name == 'motion_joint_velocities_10frame_step1':
                buf[off:off + dim] = self._gather_motion_jvel(10, 1)
            elif name == 'motion_root_z_position':
                buf[off:off + dim] = self._gather_root_z(1, 1)
            elif name == 'motion_root_z_position_10frame_step5':
                buf[off:off + dim] = self._gather_root_z(10, 5)
            else:
                raise NotImplementedError(
                    f'observation {name!r} required by mode {self.encode_mode} '
                    'has no gatherer in this port')
        return buf

    def _policy_observations(self):
        buf = np.zeros(self.dec_dim)
        for name, off, dim in self.policy_obs:
            if name == 'token_state':
                buf[off:off + dim] = self.token_state
            elif name == 'his_base_angular_velocity_10frame_step1':
                buf[off:off + dim] = self._gather_his('base_ang_vel')
            elif name == 'his_body_joint_positions_10frame_step1':
                buf[off:off + dim] = self._gather_his('body_q')
            elif name == 'his_body_joint_velocities_10frame_step1':
                buf[off:off + dim] = self._gather_his('body_dq')
            elif name == 'his_last_actions_10frame_step1':
                buf[off:off + dim] = self._gather_his('last_action')
            elif name == 'his_gravity_dir_10frame_step1':
                buf[off:off + dim] = self._gather_gravity_dir()
            else:
                raise NotImplementedError(f'policy observation {name!r}')
        return buf

    # -- one 50 Hz control tick ---------------------------------------------------
    def step(self, base_quat, base_ang_vel, q_mj, dq_mj):
        """base_quat: pelvis wxyz (world); base_ang_vel: pelvis ang vel in
        the BASE frame (gyro convention); q_mj/dq_mj: 29 joints, MuJoCo
        order.  Returns PD target joint positions in MuJoCo order."""
        base_quat = np.asarray(base_quat, float)

        # 1. log state (with last tick's action) -- GatherRobotStateToLogger
        q_isaac = np.asarray(q_mj, float)[P.MUJOCO_TO_ISAACLAB]
        dq_isaac = np.asarray(dq_mj, float)[P.MUJOCO_TO_ISAACLAB]
        self.hist.append(_Entry(base_quat.copy(),
                                np.asarray(base_ang_vel, float).copy(),
                                q_isaac - P.DEFAULT_ANGLES_ISAAC, dq_isaac,
                                self.last_action.copy()))
        if len(self.hist) > 4 * _HIS_LEN:
            del self.hist[:-2 * _HIS_LEN]

        # 2. heading state -- UpdateHeadingState
        self._update_heading(base_quat)

        # 3. encoder -> token_state -- GatherTokenState
        enc_obs = self._encoder_observations(base_quat)
        token = self.enc.run(None, {self.enc_in:
                                    enc_obs[None].astype(np.float32)})[0]
        self.token_state = token[0].astype(float)

        # 4. decoder -> action -- CreatePolicyCommand
        dec_obs = self._policy_observations()
        action = self.dec.run(None, {self.dec_in:
                                     dec_obs[None].astype(np.float32)})[0][0]
        self.last_action = action.astype(float)
        q_target = P.DEFAULT_ANGLES + \
            self.last_action[P.ISAACLAB_TO_MUJOCO] * P.ACTION_SCALE

        # 5. frame advancement -- CurrentFrameAdvancement
        if self.play:
            self.current_frame += 1
            if self.current_frame >= self.motion.timesteps:
                if self.streaming:
                    self.current_frame = self.motion.timesteps - 1
                else:
                    self.play = False
                    self.current_frame = 0
                    self.reinitialize_heading = True

        return q_target
