"""Variant (b): physical box + weld attach. While the reference says the box
is held AND the physical palms are near the physical box, a weld equality
drives the box to the reference's box-in-pelvis pose relative to the PHYSICAL
pelvis -- so the box's weight genuinely loads the tracked robot, and release
hands it back to physics.
"""
import sys

from demo_base import Demo, run_main


class WeldDemo(Demo):
    MODE = 'weld'

    def _sync_box(self, f):
        d, m = self.data, self.model
        eq = self.ids['weld_eq']
        if self.ref_attached(f):
            if not self.attached:
                if self.palm_box_dist() > self.args.engage_dist:
                    return
                self.attached = True
                d.eq_active[eq] = 1
                print(f'[{self.t:6.2f}s] weld ENGAGED '
                      f'(palm-box {self.palm_box_dist():.2f} m)')
            rel_p, rel_q = self.ref_box_relpose(f)
            m.eq_data[eq, 0:3] = 0.0
            m.eq_data[eq, 3:6] = rel_p
            m.eq_data[eq, 6:10] = rel_q
        elif self.attached:
            self.attached = False
            d.eq_active[eq] = 0
            print(f'[{self.t:6.2f}s] weld RELEASED')


def extra_args(ap):
    ap.add_argument('--engage-dist', type=float, default=0.45,
                    help='max palm-to-box-centre distance to engage the weld')


if __name__ == '__main__':
    sys.exit(run_main(WeldDemo, 'mm_sonic_box_weld.mp4', extra_args))
