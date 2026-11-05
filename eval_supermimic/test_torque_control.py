"""Check clipping order against actual MuJoCo joint actuator torque."""
import sys
from pathlib import Path
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sonic_g1_box.torque_control import compose_torque, configure_torque_clipping


class Policy:
    torque_limits = np.tile([-10., 10.], (29, 1))

    def clip_torque(self, torque):
        return np.clip(torque, self.torque_limits[:, 0], self.torque_limits[:, 1])


class TorqueControlTest(unittest.TestCase):
    def model(self):
        bodies = ''.join(f'<body pos="{i} 0 0"><joint name="j{i}" type="hinge" '
                         'actuatorfrclimited="true" actuatorfrcrange="-8 8"/>'
                         '<geom type="sphere" size="0.1" mass="1"/></body>' for i in range(29))
        motors = ''.join(f'<motor joint="j{i}" ctrllimited="true" ctrlrange="-12 12" '
                         'forcelimited="true" forcerange="-11 11"/>' for i in range(29))
        return mujoco.MjModel.from_xml_string(
            f'<mujoco><option gravity="0 0 0"/><worldbody>{bodies}</worldbody>'
            f'<actuator>{motors}</actuator></mujoco>')

    def applied(self, order, extra):
        model = self.model()
        data = mujoco.MjData(model)
        clip = configure_torque_clipping(model, Policy(), 'supermimic', order)
        data.ctrl[:] = compose_torque(np.full(29, 15.), extra, clip, order)
        mujoco.mj_step(model, data)
        return data.qfrc_actuator.copy()

    def test_after_force_respects_scene_joint_limit(self):
        np.testing.assert_array_equal(self.applied('after-force', np.full(29, 35.)), 8.)

    def test_before_force_adds_beyond_all_scene_limits(self):
        np.testing.assert_array_equal(self.applied('before-force', np.full(29, 35.)), 43.)

    def test_no_virtual_force_preserves_applied_pd(self):
        np.testing.assert_array_equal(self.applied('after-force', None),
                                      self.applied('before-force', None))

    def test_negative_force(self):
        clip = Policy().clip_torque
        np.testing.assert_array_equal(compose_torque(np.full(29, -15.), np.full(29, -35.),
                                                     clip, 'before-force'), -45.)

    def test_sonic_keeps_scene_limits(self):
        model = self.model()
        before = model.jnt_actfrclimited.copy(), model.actuator_ctrllimited.copy()
        self.assertIsNone(configure_torque_clipping(model, None, 'sonic', 'after-force'))
        np.testing.assert_array_equal(model.jnt_actfrclimited, before[0])
        np.testing.assert_array_equal(model.actuator_ctrllimited, before[1])


if __name__ == '__main__':
    unittest.main()
