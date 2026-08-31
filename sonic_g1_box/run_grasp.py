"""Variant (c): true frictional grasp. Nothing attaches the box -- only
contact friction between the palm pads (and forearms) and the box can lift and
hold it. The reference hands merely touch the box surface, so tracking alone
produces no squeeze pressure. While a hand's SceneBot contact label is on at
the tracked frame, --shoulder-squeeze biases that shoulder-roll PD target
inward (palm pressure) and --wrist-squeeze toes the wrist yaw in (fingers
press toward the box); --arm-kp / --arm-kd scale the arm PD so the biases
become real force.
"""
import sys

from demo_base import Demo, run_main

# MuJoCo-order arm indices -- left 15..21 / right 22..28, per arm:
#   +0 shoulder_pitch  +1 shoulder_roll  +2 shoulder_yaw  +3 elbow
#   +4 wrist_roll      +5 wrist_pitch    +6 wrist_yaw
# Scale a subgroup in setup_extra with e.g. `self.kps[WRISTS] *= 2.0`.
ARM_JOINTS = slice(15, 29)
SHOULDERS = [15, 16, 17, 22, 23, 24]
ELBOWS = [18, 25]
WRISTS = [19, 20, 21, 26, 27, 28]
L_SHOULDER_ROLL, R_SHOULDER_ROLL = 16, 23
# Inward = same mirroring as the shoulder rolls (measured in the grab pose:
# left yaw NEGATIVE / right yaw POSITIVE toe the palms toward the box).
L_WRIST_YAW, R_WRIST_YAW = 21, 28


class GraspDemo(Demo):
    MODE = 'grasp'

    def setup_extra(self):
        self.kps[ARM_JOINTS] *= self.args.arm_kp
        self.kds[ARM_JOINTS] *= self.args.arm_kd

    def _adjust_target(self, target, f):
        lc, rc = self.motion.meta_at(f)[4]
        if lc or rc:
            target = target.copy()
            if lc:
                target[L_SHOULDER_ROLL] -= self.args.shoulder_squeeze
                target[L_WRIST_YAW] -= self.args.wrist_squeeze
            if rc:
                target[R_SHOULDER_ROLL] += self.args.shoulder_squeeze
                target[R_WRIST_YAW] += self.args.wrist_squeeze
        return target


def extra_args(ap):
    ap.add_argument('--arm-kp', type=float, default=1.0,
                    help='scale on the arm PD stiffness (squeeze strength)')
    ap.add_argument('--arm-kd', type=float, default=1.0,
                    help='scale on the arm PD damping')
    ap.add_argument('--shoulder-squeeze', type=float, default=0.2,
                    help='inward shoulder-roll bias (rad) while that '
                         "hand's contact label is on (palm pressure)")
    ap.add_argument('--wrist-squeeze', type=float, default=0.0,
                    help="inward wrist-yaw bias (rad) while that hand's "
                         'label is on (toes the palm into the box)')
    ap.set_defaults(max_seconds=60.0)


if __name__ == '__main__':
    sys.exit(run_main(GraspDemo, 'mm_sonic_box_grasp.mp4', extra_args))
