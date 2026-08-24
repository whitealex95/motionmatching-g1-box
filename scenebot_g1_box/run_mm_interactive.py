"""Interactive physical demo: drive the motion-matched G1 with the keyboard
while the SceneBot policy tracks it in full MuJoCo physics.

Same pipeline as run_mm_pickup (open loop: the keyboard steers the REFERENCE
matcher, the policy follows through the anchor-error observation), but the
scripted commander is replaced by live key state in a GLFW window, and the
reference stream runs only 2 frames ahead of the playhead so key response is
immediate (the scripted runner buffers 0.4 s ahead).

    W / A / S / D    move (relative to the camera); Shift = walk pace
    Arrow keys       face direction, independent of travel
    B                box action: walk over + pick up, or set down while carrying
    T                toggle gizmos (route, command trajectory, contact prompts)
    Esc              quit             (no reset key: restart the script)

--script replays a fixed key timeline headlessly instead (settle, B, carry-
walk 3 s, B, retreat) and exits 0/1 -- the regression test for this runner.
"""
import argparse
import math
import os
import sys
import time

if '--script' not in sys.argv:
    os.environ.setdefault('MUJOCO_GL', 'glfw')

import numpy as np

from run_mm_pickup import Demo, MMPacketAdapter  # noqa: F401 (env setup too)

import mujoco

import contact_viz
from mm_g1 import config as C
from scenebot_tracking import params as P

_TRAJ_RGBA = np.array([0.9, 0.1, 0.1, 1.0], np.float32)
_MARK_RGBA = np.array([0.15, 0.8, 0.25, 0.9], np.float32)
_EYE3 = np.eye(3).ravel()
INTERACTIVE_MARGIN = 2            # stream frames ahead of the playhead
MAX_SPEED = 1.2                   # full-stick command (m/s); policy-safe
WALK_SCALE = 0.4


class InteractiveDemo(Demo):
    def __init__(self, args):
        self.script = args.script
        self.held = set()
        self.shift = False
        self.cam_azimuth = -35.0
        super().__init__(args)
        self.margin = INTERACTIVE_MARGIN

    # --- commander: live keys (or the --script timeline) ---------------------
    def _command(self, mm):
        # The matcher's box belief is snapped to the PHYSICAL box whenever the
        # box is not held (resting, being walked to, reached for, or released)
        # -- so a kicked or moved box is approached and picked where it really
        # is. While held (through the carry) the reference box is authoritative
        # and rides the robot. Open loop shares one world frame with physics,
        # so the physical pose is valid in the matcher frame.
        if self.mode != 'kinematic' and not mm.box_held:
            mm.boxPos[:] = self.data.qpos[self.bq:self.bq + 3]
            mm.boxRot[:] = self.data.qpos[self.bq + 3:self.bq + 7]
            mm.boxPosPrev[:] = mm.boxPos
            mm.boxVelWorld[:] = 0.0
        st = mm.state_name()
        if self._mm_prev == 'PLACE' and st == 'LOCOMOTION':
            self.place_done = True
            self._place_time = self.t
        self._mm_prev = st
        if self.script:
            return self._script_command(mm, st)
        if st in ('PICK', 'PLACE'):
            return np.zeros(3), np.zeros(3)
        return self._keys_command(mm, st)

    def _keys_command(self, mm, st):
        import glfw

        fwd = math.radians(self.cam_azimuth)
        right = fwd - math.pi / 2.0
        fdir = np.array([math.cos(fwd), math.sin(fwd), 0.0])
        rdir = np.array([math.cos(right), math.sin(right), 0.0])
        move = np.zeros(3)
        if glfw.KEY_W in self.held: move += fdir
        if glfw.KEY_S in self.held: move -= fdir
        if glfw.KEY_D in self.held: move += rdir
        if glfw.KEY_A in self.held: move -= rdir
        face = np.zeros(3)
        if glfw.KEY_UP in self.held:    face += fdir
        if glfw.KEY_DOWN in self.held:  face -= fdir
        if glfw.KEY_RIGHT in self.held: face += rdir
        if glfw.KEY_LEFT in self.held:  face -= rdir
        m = float(np.linalg.norm(move))
        if m > 1e-6:
            top = C.CARRY_MAX_SPEED if st == 'CARRY' else MAX_SPEED
            move = move / m * (top * (WALK_SCALE if self.shift else 1.0))
        else:
            move = np.zeros(3)
        f = float(np.linalg.norm(face))
        face = face / f if f > 1e-6 else np.zeros(3)
        return move, face

    def _script_command(self, mm, st):
        if st in ('PICK', 'PLACE', 'MOVE-TO-PICK'):
            return np.zeros(3), np.zeros(3)
        if st == 'CARRY':
            if self.carry_start is None:
                self.carry_start = self.t
            elif self.t - self.carry_start > self.args.carry_seconds:
                mm.trigger_box()
            return np.array([0.4, 0.0, 0.0]), np.zeros(3)   # carry-walk +x
        if self.place_done:
            if self.t - self._place_time < self.args.post_place_pause:
                return np.zeros(3), np.zeros(3)
            away = self.motion._mm_t1[0:2] - mm.boxPos[0:2]
            u = away / max(float(np.linalg.norm(away)), 1e-6)
            return np.array([0.6 * u[0], 0.6 * u[1], 0.0]), np.zeros(3)
        if self.t >= 1.0 and not self.pick_triggered:
            mm.trigger_box()
            self.pick_triggered = True
        return np.zeros(3), np.zeros(3)

    # --- headless script run (the regression test) ---------------------------
    def run_script(self):
        wall = time.time()
        for tick in range(int(self.args.max_seconds * 1.0 / P.CONTROL_DT)):
            self.t = tick * P.CONTROL_DT
            self.step_control()
            self._check_fall()
            self.render()
            if self.fallen and (self.t - self.fall_time) > 2.0:
                break
            if self.place_done and self._robot_box_dist() > self.WALK_AWAY_DIST:
                break
        return self.finish(wall)

    # --- interactive window --------------------------------------------------
    def run_window(self):
        import glfw

        if not glfw.init():
            raise RuntimeError('glfw.init() failed -- this needs a display '
                               '(run with a desktop / X forwarding)')
        window = glfw.create_window(
            1280, 720, 'mm + SceneBot physical -- WASD move, B box', None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError('no GLFW window (no display?)')
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = self.cam_azimuth, -18.0, 3.6
        cam.lookat[:] = [0.0, 0.0, 0.75]
        opt = mujoco.MjvOption()
        scene = mujoco.MjvScene(self.model, maxgeom=10000)
        ctx = mujoco.MjrContext(self.model,
                                mujoco.mjtFontScale.mjFONTSCALE_150)
        show_gizmos = True
        mouse = {'last': None, 'left': False, 'right': False}

        def on_key(win, key, scancode, action, mods):
            nonlocal show_gizmos
            import glfw as g
            self.shift = bool(mods & g.MOD_SHIFT)
            if action == g.PRESS:
                if key == g.KEY_ESCAPE:
                    g.set_window_should_close(win, True)
                elif key == g.KEY_B:
                    self.matcher.trigger_box()
                elif key == g.KEY_T:
                    show_gizmos = not show_gizmos
                else:
                    self.held.add(key)
            elif action == g.RELEASE:
                self.held.discard(key)

        def on_button(win, button, action, mods):
            import glfw as g
            press = action == g.PRESS
            if button == g.MOUSE_BUTTON_LEFT:
                mouse['left'] = press
            elif button == g.MOUSE_BUTTON_RIGHT:
                mouse['right'] = press
            mouse['last'] = g.get_cursor_pos(win) if press else None

        def on_cursor(win, x, y):
            if mouse['last'] is None:
                return
            dx, dy = x - mouse['last'][0], y - mouse['last'][1]
            mouse['last'] = (x, y)
            _, h = glfw.get_window_size(win)
            act = (mujoco.mjtMouse.mjMOUSE_ROTATE_V if mouse['left'] else
                   mujoco.mjtMouse.mjMOUSE_MOVE_V if mouse['right'] else None)
            if act is not None:
                mujoco.mjv_moveCamera(self.model, act, dx / h, dy / h,
                                      scene, cam)

        def on_scroll(win, xo, yo):
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                                  0.0, -0.05 * yo, scene, cam)

        glfw.set_key_callback(window, on_key)
        glfw.set_mouse_button_callback(window, on_button)
        glfw.set_cursor_pos_callback(window, on_cursor)
        glfw.set_scroll_callback(window, on_scroll)

        last = glfw.get_time()
        acc = 0.0
        while not glfw.window_should_close(window):
            now = glfw.get_time()
            acc = min(acc + now - last, 0.2)   # never spiral after a stall
            last = now
            while acc >= P.CONTROL_DT:
                self.cam_azimuth = float(cam.azimuth)
                self.t += P.CONTROL_DT
                self.step_control()
                self._check_fall()
                acc -= P.CONTROL_DT

            cam.lookat[0] = float(self.data.qpos[self.rq])
            cam.lookat[1] = float(self.data.qpos[self.rq + 1])
            w, h = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, w, h)
            mujoco.mjv_updateScene(self.model, self.data, opt, None, cam,
                                   mujoco.mjtCatBit.mjCAT_ALL, scene)
            self._draw_ghost(scene)
            if show_gizmos:
                contact_viz.draw(scene, self.data, self.contact_sites,
                                 self.policy.contact_mask)
                self._draw_gizmos(scene)
            mujoco.mjr_render(viewport, scene, ctx)
            self._overlay(viewport, ctx)
            glfw.swap_buffers(window)
            glfw.poll_events()
        glfw.terminate()
        return 0

    def _draw_gizmos(self, scn):
        m = self.matcher
        for (px, py, _), (dx, dy, _) in zip(m.Tpos, m.Tdir):
            self._stick(scn, [px, py, 0.05],
                        [px + 0.25 * dx, py + 0.25 * dy, 0.05], _TRAJ_RGBA)
        if m.state_name() == 'MOVE-TO-PICK':
            for a, b in zip(m.route_pts[:-1], m.route_pts[1:]):
                self._stick(scn, [a[0], a[1], 0.05], [b[0], b[1], 0.05],
                            _MARK_RGBA)

    @staticmethod
    def _stick(scn, p0, p1, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), _EYE3, rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012,
                             np.asarray(p0, float), np.asarray(p1, float))
        scn.ngeom += 1

    def _overlay(self, viewport, ctx):
        st = self.matcher.state_name()
        head = {'LOCOMOTION': 'LOCOMOTION  [B: walk over + pick up]',
                'MOVE-TO-PICK': 'WALKING TO THE BOX  [B: cancel]',
                'CARRY': 'CARRY  [B: set down]'}.get(st, st)
        box_z = float(self.data.qpos[self.bq + 2])
        body = (f'mode: {self.mode}   box z {box_z:.2f} m'
                f'{"   FELL" if self.fallen else ""}\n'
                'WASD move | arrows face | Shift walk | B box | T gizmos\n'
                'drag orbit | right-drag pan | scroll zoom | Esc quit')
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                           mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport,
                           head, body, ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['kinematic', 'grasp'], default='grasp')
    ap.add_argument('--script', action='store_true',
                    help='headless scripted run instead of the window')
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--max-seconds', type=float, default=50.0)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--carry-seconds', type=float, default=3.0)
    ap.add_argument('--post-place-pause', type=float, default=2.0)
    ap.add_argument('--arm-gain', type=float, default=1.0)
    ap.add_argument('--squeeze', type=float, default=0.0)
    ap.add_argument('--box-mass', type=float, default=0.1)
    ap.add_argument('--box-size', type=float, nargs=3,
                    default=list(C.BOX_HALF), metavar=('X', 'Y', 'Z'))
    ap.add_argument('--box-scale', type=float, default=1.0)
    ap.add_argument('--video', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'out',
        'scenebot_mm_interactive.mp4'))
    args = ap.parse_args()
    args.video_path = args.video
    args.viewer = False
    args.ref_mode = 'none'
    args.settle_seconds = 1.0
    args.retry_cooldown = 2.0
    args.anchor_gain = 0.2
    args.replan_gain = 0.738
    if not args.script:
        args.no_video = True          # the window IS the output
    demo = InteractiveDemo(args)
    return demo.run_script() if args.script else demo.run_window()


if __name__ == '__main__':
    sys.exit(main())
