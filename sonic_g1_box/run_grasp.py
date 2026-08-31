"""Variant (c): true frictional grasp. Nothing attaches the box -- only
contact friction between the palm pads (and forearms) and the box can lift and
hold it. The reference hands merely touch the box surface, so tracking alone
produces no squeeze pressure. The SceneBot contact labels at the tracked
frame drive a single open/close mode for BOTH hands: while either label is
on, --shoulder-squeeze presses the shoulder rolls in and --wrist-squeeze
toes the wrist yaws in; while both are off during the PICK/PLACE ride,
--shoulder-open swings the shoulder rolls out to clear the box.
--arm-kp / --arm-kd scale the arm PD so the biases become real force.
"""
import sys

import numpy as np

from demo_base import Demo, run_main
from mm_g1.states import State

# MuJoCo-order arm indices -- left 15..21 / right 22..28, per arm:
#   +0 shoulder_pitch  +1 shoulder_roll  +2 shoulder_yaw  +3 elbow
#   +4 wrist_roll      +5 wrist_pitch    +6 wrist_yaw
# Scale a subgroup in setup_extra with e.g. `self.kps[WRISTS] *= 2.0`.
ARM_JOINTS = slice(15, 29)
SHOULDERS = [15, 16, 17, 22, 23, 24]
ELBOWS = [18, 25]
WRISTS = [19, 20, 21, 26, 27, 28]
L_SHOULDER_ROLL, R_SHOULDER_ROLL = 16, 23
L_WRIST_YAW, R_WRIST_YAW = 21, 28
CUSTOM_ARM_JOINTS = [16, 23, 21, 28]


class GraspDemo(Demo):
    MODE = 'grasp'

    def setup_extra(self):
        self.kps[CUSTOM_ARM_JOINTS] *= self.args.arm_kp
        self.kds[CUSTOM_ARM_JOINTS] *= self.args.arm_kd
        self._palm_sites = [self.model.site('left_palm').id,
                            self.model.site('right_palm').id]
        self._arm_idx = [list(range(15, 22)), list(range(22, 29))]
        self._palm_ramp = [0.0, 0.0]

    def _ctrl_extra(self, f):
        """Jacobian squeeze: while a hand's label is on, press its palm toward
        the box centre with --palm-force newtons (tau = J^T f on that arm),
        ramped over 0.15 s so contact makes and releases without a step."""
        F = self.args.palm_force
        if F <= 0.0:
            return None
        import mujoco
        _, _, _, _, (lc, rc) = self.motion.meta_at(f)
        tau = np.zeros(29)
        box_c = self.data.qpos[self.bq:self.bq + 3]
        jacp = np.zeros((3, self.model.nv))
        for h, on in enumerate((lc, rc)):
            step = 0.02 / 0.15
            self._palm_ramp[h] = (min(1.0, self._palm_ramp[h] + step) if on
                                  else max(0.0, self._palm_ramp[h] - step))
            if self._palm_ramp[h] <= 0.0:
                continue
            sid = self._palm_sites[h]
            mujoco.mj_jacSite(self.model, self.data, jacp, None, sid)
            n = box_c - self.data.site_xpos[sid]
            L = float(np.linalg.norm(n))
            if L < 1e-6:
                continue
            fvec = (self._palm_ramp[h] * F / L) * n
            for j in self._arm_idx[h]:
                tau[j] += float(jacp[:, self.dq_at[j]] @ fvec)
        return tau

    def _adjust_target(self, target, f):
        _, _, state, _, (lc, rc) = self.motion.meta_at(f)
        target = target.copy()
        if lc or rc:                              # closing: squeeze both hands
            target[L_SHOULDER_ROLL] -= self.args.shoulder_squeeze
            target[R_SHOULDER_ROLL] += self.args.shoulder_squeeze
            target[L_WRIST_YAW] -= self.args.wrist_squeeze
            target[R_WRIST_YAW] += self.args.wrist_squeeze
        elif state in (State.PICK, State.PLACE):  # opening: clear the box
            target[L_SHOULDER_ROLL] += self.args.shoulder_open
            target[R_SHOULDER_ROLL] -= self.args.shoulder_open
        return target


def extra_args(ap):
    ap.add_argument('--arm-kp', type=float, default=1.0,
                    help='scale on the arm PD stiffness (squeeze strength)')
    ap.add_argument('--arm-kd', type=float, default=1.0,
                    help='scale on the arm PD damping')
    ap.add_argument('--shoulder-open', type=float, default=0.0,
                    help='outward shoulder-roll bias (rad), both hands, during '
                         'PICK/PLACE while both contact labels are off')
    ap.add_argument('--shoulder-squeeze', type=float, default=0.0,
                    help='inward shoulder-roll bias (rad), both hands, while '
                         'either contact label is on (palm pressure)')
    ap.add_argument('--palm-force', type=float, default=0.0,
                    help='Jacobian squeeze: newtons pressing each palm toward '
                         "the box centre while that hand's label is on")
    ap.add_argument('--wrist-squeeze', type=float, default=0.0,
                    help='inward wrist-yaw bias (rad), both hands, while '
                         'either label is on (toes the palms into the box)')
    ap.set_defaults(max_seconds=60.0)


if __name__ == '__main__':
    sys.exit(run_main(GraspDemo, 'mm_sonic_box_grasp.mp4', extra_args))
