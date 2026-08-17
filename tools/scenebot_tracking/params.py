"""Paths and policy_meta constants for the vendored SceneBot bundle."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET_DIR = os.path.join(ROOT, 'assets', 'scenebot')

POLICY_ONNX = os.path.join(ASSET_DIR, 'policy.onnx')
POLICY_META = os.path.join(ASSET_DIR, 'policy_meta.json')
CLIPS_BIN = os.path.join(ASSET_DIR, 'clips.bin')
CLIPS_INDEX = os.path.join(ASSET_DIR, 'clips_index.json')
CONTACT_BIN = os.path.join(ASSET_DIR, 'contact_labels.bin')
CONTACT_INDEX = os.path.join(ASSET_DIR, 'contact_labels_index.json')
MOTION_GRAPH = os.path.join(ASSET_DIR, 'motion_graph.json')
SCENE_XML = os.path.join(ASSET_DIR, 'scene_29dof_flat_hand.xml')
SCENE_FLOOR_XML = os.path.join(ASSET_DIR, 'scene_box_floor.xml')

with open(POLICY_META) as f:
    META = json.load(f)

CONTROL_DT = float(META['control_dt'])          # 0.02 -> 50 Hz policy
SIM_DT = float(META['simulation_dt'])           # 0.005 -> 200 Hz physics
DECIMATION = max(1, round(CONTROL_DT / SIM_DT))

# index maps as used by the demo: ISAAC_TO_MUJOCO[mj] = isaac index of
# mujoco joint mj; MUJOCO_TO_ISAAC[isaac] = mujoco index of isaac joint.
ISAAC_TO_MUJOCO = np.array(META['ISAAC_TO_MUJOCO'], int)
MUJOCO_TO_ISAAC = np.array(META['MUJOCO_TO_ISAAC'], int)

ACTION_SCALE_MJ = np.array(META['action_scale_mujoco'], float)   # (29,) mj order
DEFAULT_Q_ISAAC = np.array(META['default_q_isaac'], float)       # (29,) isaac order
KPS = np.array(META['joint_stiffness'], float)                   # (29,) mj order
KDS = np.array(META['joint_damping'], float)                     # (29,) mj order
TORQUE_LIMIT = np.array(META['torque_limit'], float)             # (29,) mj order
INIT_QPOS_36 = np.array(META['init_qpos_36'], float)

OBS_DIM = int(META['obs_dim'])
ACTION_DIM = int(META['action_dim'])
OBS_NAMES = list(META['obs_names'])
