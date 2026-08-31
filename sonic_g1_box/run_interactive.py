"""Interactive SONIC demo -- drive the PHYSICS robot like run.py.

The same stack as run_grasp (matcher streamed as the SONIC reference, full
physics, frictional box, label-gated squeeze), but the commander is YOUR
keyboard instead of the script, in a GLFW window with true held-key input:

  W / A / S / D    move, relative to the camera
  Arrow keys       face direction, independent of travel
  Shift (hold)     walk instead of run
  B                box action: pick up (walks over by itself) / set down
  Left-drag        orbit camera     Right-drag  pan     Scroll  zoom
  Esc              quit

    python run_interactive.py [--sonic ...] [--shoulder-squeeze ...] ...
"""
import math
import os
import sys

os.environ['MUJOCO_GL'] = 'glfw'
import numpy as np
import glfw
import mujoco

from demo_base import build_argparser
from run_grasp import GraspDemo, extra_args
from mm_g1 import config as C
from mm_g1.states import State
from sonic_tracking import params as P

_MOVE = {glfw.KEY_W, glfw.KEY_A, glfw.KEY_S, glfw.KEY_D}
_FACE = {glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT}


class InputState:
    """Held keys -> (desiredVel, desiredFace), camera-relative like run.py."""

    def __init__(self):
        self.held = set()
        self.shift = False
        self.cam = None                              # set by App

    def command(self, carry=False):
        if self.cam is None:
            return np.zeros(3), np.zeros(3)
        fwd = math.radians(self.cam.azimuth)
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
            top = C.CARRY_MAX_SPEED if carry else C.MAX_SPEED
            if self.shift:
                top *= C.WALK_SCALE
            move = move / m * top
        f = float(np.linalg.norm(face))
        if f > 1e-6:
            face = face / f
        return move, face


class InteractiveGrasp(GraspDemo):
    """run_grasp's physics + belief feedback + grip check, keyboard commands."""

    def __init__(self, args, inp):
        self.inp = inp
        super().__init__(args)

    def _command(self, mm):
        d = self.data
        if not mm.box_held:
            mm.boxPos[:] = d.qpos[self.bq:self.bq + 3]
            mm.boxRot[:] = d.qpos[self.bq + 3:self.bq + 7]
        else:
            if not self._prev_mm_held:
                self._hold_start_f = self.motion.timesteps
            self._check_grip(mm)
        self._prev_mm_held = mm.box_held
        return self.inp.command(carry=mm.state is State.CARRY)


class App:
    def __init__(self, demo, inp, width=1280, height=720):
        self.demo, self.inp = demo, inp
        if not glfw.init():
            raise RuntimeError('glfw.init() failed -- needs a display')
        self.window = glfw.create_window(
            width, height, 'SONIC interactive -- WASD move, B box', None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.8, 120.0, -18.0
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(demo.model, maxgeom=10000)
        self.ctx = mujoco.MjrContext(demo.model,
                                     mujoco.mjtFontScale.mjFONTSCALE_150)
        inp.cam = self.cam
        self.look = demo._focus_point()
        self.cam.lookat[:] = self.look
        self._mouse_last = None
        self._button = {'left': False, 'right': False}
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    def _on_key(self, window, key, scancode, action, mods):
        self.inp.shift = bool(mods & glfw.MOD_SHIFT)
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_B:
                self.demo.matcher.trigger_box()
            elif key in _MOVE or key in _FACE:
                self.inp.held.add(key)
        elif action == glfw.RELEASE:
            self.inp.held.discard(key)

    def _on_mouse_button(self, window, button, action, mods):
        press = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._button['left'] = press
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._button['right'] = press
        self._mouse_last = glfw.get_cursor_pos(window) if press else None

    def _on_cursor(self, window, x, y):
        if self._mouse_last is None:
            return
        dx, dy = x - self._mouse_last[0], y - self._mouse_last[1]
        self._mouse_last = (x, y)
        w, h = glfw.get_window_size(window)
        if self._button['left']:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif self._button['right']:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            return
        mujoco.mjv_moveCamera(self.demo.model, action, dx / h, dy / h,
                              self.scene, self.cam)

    def _on_scroll(self, window, xoff, yoff):
        mujoco.mjv_moveCamera(self.demo.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                              0.0, -0.05 * yoff, self.scene, self.cam)

    def run(self, max_frames=None):
        demo = self.demo
        last = glfw.get_time()
        acc = 0.0
        frames = 0
        while not glfw.window_should_close(self.window):
            now = glfw.get_time()
            acc = min(acc + now - last, 4 * P.CONTROL_DT)
            last = now
            while acc >= P.CONTROL_DT:
                demo.step_physics()
                demo.t += P.CONTROL_DT
                acc -= P.CONTROL_DT
            self.look += 0.06 * (demo._focus_point() - self.look)
            self.cam.lookat[:] = self.look
            w, h = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, w, h)
            mujoco.mjv_updateScene(demo.model, demo.data, self.opt, None,
                                   self.cam, mujoco.mjtCatBit.mjCAT_ALL,
                                   self.scene)
            demo._draw_ghost(self.scene)
            mujoco.mjr_render(viewport, self.scene, self.ctx)
            title, body = demo._overlay_text()
            body += ('\nWASD move | arrows face | Shift walk | B box | '
                     'Esc quit')
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                               mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport,
                               title, body, self.ctx)
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
        glfw.terminate()


def main():
    ap = build_argparser('interactive.mp4')
    extra_args(ap)
    ap.add_argument('--smoke-frames', type=int, default=None,
                    help='(testing) exit after N rendered frames')
    args = ap.parse_args()
    args.no_video, args.viewer = True, False
    args.video_path = args.video
    inp = InputState()
    demo = InteractiveGrasp(args, inp)
    app = App(demo, inp)
    print('Window open -- WASD to move, B to pick up / set down, Esc to quit.')
    app.run(max_frames=args.smoke_frames)
    return 0


if __name__ == '__main__':
    sys.exit(main())
