"""Variant (c): true frictional grasp. Nothing attaches the box -- only
contact friction between the palm pads (and forearms) and the box can lift and
hold it. The reference hands merely touch the box surface, so tracking alone
produces no squeeze pressure. While a hand's SceneBot contact label is on at
the tracked frame, --shoulder-squeeze biases that shoulder-roll PD target
inward (palm pressure) and --elbow-squeeze adds elbow flexion (hugs the box
toward the chest); --arm-gain stiffens the arms so the biases become real
force.
"""
import sys

import numpy as np

from demo_base import Demo, run_main

ARM_JOINTS = slice(15, 29)     # shoulders..wrists in MuJoCo order
L_SHOULDER_ROLL, R_SHOULDER_ROLL = 16, 23
L_ELBOW, R_ELBOW = 18, 25      # positive = flexion (hands toward the chest)


class GraspDemo(Demo):
    MODE = 'grasp'

    def setup_extra(self):
        self.kps[ARM_JOINTS] *= self.args.arm_gain
        self.kds[ARM_JOINTS] *= np.sqrt(self.args.arm_gain)

    def _adjust_target(self, target, f):
        lc, rc = self.motion.meta_at(f)[4]
        if lc or rc:
            target = target.copy()
            if lc:
                target[L_SHOULDER_ROLL] -= self.args.shoulder_squeeze
                target[L_ELBOW] += self.args.elbow_squeeze
            if rc:
                target[R_SHOULDER_ROLL] += self.args.shoulder_squeeze
                target[R_ELBOW] += self.args.elbow_squeeze
        return target


def extra_args(ap):
    ap.add_argument('--arm-gain', type=float, default=1.0,
                    help='scale on the arm PD stiffness (squeeze strength)')
    ap.add_argument('--shoulder-squeeze', type=float, default=0.2,
                    help='inward shoulder-roll bias (rad) while that '
                         "hand's contact label is on (palm pressure)")
    ap.add_argument('--elbow-squeeze', type=float, default=0.0,
                    help="extra elbow flexion (rad) while that hand's "
                         'label is on (hugs the box toward the chest)')
    ap.set_defaults(max_seconds=60.0)


if __name__ == '__main__':
    sys.exit(run_main(GraspDemo, 'mm_sonic_box_grasp.mp4', extra_args))
