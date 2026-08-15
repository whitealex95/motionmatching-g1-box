"""Variant (a): kinematic box. The tracked robot is fully physical, but the
box has no collision and is teleported to the reference box pose every
substep. Proves the pick/carry/place motions track before any box physics is
involved.
"""
import sys

from demo_base import Demo, run_main


class KinematicDemo(Demo):
    MODE = 'kinematic'

    def _post_substep(self, f):
        d = self.data
        d.qpos[self.bq:self.bq + 7] = self.motion.qpos[f][36:43]
        d.qvel[self.bd:self.bd + 6] = 0.0


if __name__ == '__main__':
    sys.exit(run_main(KinematicDemo, 'mm_sonic_box_kinematic.mp4'))
