"""G1 29-DoF constants for SONIC deployment, ported from the official C++
deploy stack (gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/
policy_parameters.hpp + robot_parameters.hpp).  All values are bit-for-bit
the formulas used there.

Two joint orderings coexist (as in the C++):
  * MuJoCo/URDF/hardware order -- the simulator, reference CSVs on disk are
    NOT in this order (they are IsaacLab order), but PD gains, action scales
    and default angles below are.
  * IsaacLab order -- the policy's action output, the reference joint_pos/
    joint_vel CSVs, and every joint-space observation.

The two gather arrays convert between them exactly like the C++ ones:
  vec_isaac = vec_mujoco[MUJOCO_TO_ISAACLAB]
  vec_mujoco = vec_isaac[ISAACLAB_TO_MUJOCO]
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Asset paths (this repo's assets/sonic/; override via env var)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
SONIC_ASSETS = os.environ.get(
    'SONIC_ASSETS', os.path.join(_REPO_ROOT, 'assets', 'sonic'))
POLICY_DIR = os.path.join(SONIC_ASSETS, 'policy')
ENCODER_ONNX = os.path.join(POLICY_DIR, 'model_encoder.onnx')
DECODER_ONNX = os.path.join(POLICY_DIR, 'model_decoder.onnx')
OBS_CONFIG_YAML = os.path.join(POLICY_DIR, 'observation_config.yaml')
G1_SCENE_XML = os.path.join(SONIC_ASSETS, 'g1', 'scene_29dof.xml')
PLANNER_ONNX = os.path.join(SONIC_ASSETS, 'planner', 'planner_sonic.onnx')

NUM_JOINTS = 29
CONTROL_DT = 0.02            # 50 Hz policy / reference-motion rate
SIM_DT = 0.005               # 200 Hz physics (SIMULATE_DT in their sim yaml)
DECIMATION = int(round(CONTROL_DT / SIM_DT))

# MuJoCo/URDF/hardware joint order (matches both their g1_29dof.xml and our
# resources/g1/g1_mocap_29dof.xml -- verified identical).
JOINT_NAMES_MUJOCO = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint',
    'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint',
    'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
]

# Gather arrays (numbers verbatim from policy_parameters.hpp; the C++ names
# are kept even though they read "backwards" -- usage is what matters):
#   isaac[i] = mujoco[MUJOCO_TO_ISAACLAB[i]]
MUJOCO_TO_ISAACLAB = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])
#   mujoco[i] = isaac[ISAACLAB_TO_MUJOCO[i]]
ISAACLAB_TO_MUJOCO = np.array([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28])

# ---------------------------------------------------------------------------
# PD gains / action scale (policy_parameters.hpp formulas)
# ---------------------------------------------------------------------------
_ARMATURE = {'5020': 0.003609725, '7520_14': 0.010177520,
             '7520_22': 0.025101925, '4010': 0.00425}
_EFFORT = {'5020': 25.0, '7520_14': 88.0, '7520_22': 139.0, '4010': 5.0}
_OMEGA = 10 * 2.0 * 3.1415926535        # 10 Hz natural frequency
_ZETA = 2.0                             # damping ratio

_STIFF = {k: a * _OMEGA * _OMEGA for k, a in _ARMATURE.items()}
_DAMP = {k: 2.0 * _ZETA * a * _OMEGA for k, a in _ARMATURE.items()}

# per-joint motor type and ankle/waist-roll/pitch 2x factor, MuJoCo order
_MOTOR = ['7520_22', '7520_22', '7520_14', '7520_22', '5020', '5020',
          '7520_22', '7520_22', '7520_14', '7520_22', '5020', '5020',
          '7520_14', '5020', '5020',
          '5020', '5020', '5020', '5020', '5020', '4010', '4010',
          '5020', '5020', '5020', '5020', '5020', '4010', '4010']
_GAIN2X = [False] * 4 + [True, True] + [False] * 4 + [True, True] + \
          [False, True, True] + [False] * 14

KPS = np.array([(2.0 if d else 1.0) * _STIFF[m] for m, d in zip(_MOTOR, _GAIN2X)])
KDS = np.array([(2.0 if d else 1.0) * _DAMP[m] for m, d in zip(_MOTOR, _GAIN2X)])
# action scale uses the UN-doubled stiffness for every joint (as in the C++)
ACTION_SCALE = np.array([0.25 * _EFFORT[m] / _STIFF[m] for m in _MOTOR])

# Default standing pose (MuJoCo order)
DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,       # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,       # right leg
    0.0, 0.0, 0.0,                              # waist
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,          # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,         # right arm
])
DEFAULT_ANGLES_ISAAC = DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB]

# Standing pelvis height for that pose (their XML pelvis @0.793 is the
# zero-pose height; with knees bent ~0.669 the standing root sits lower).
DEFAULT_ROOT_Z = 0.76
