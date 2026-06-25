"""Interactive GLFW + MuJoCo viewer: drive the G1 with the keyboard in real time.

We open our own GLFW window (rather than mujoco.viewer) so we get true held-key state
from the GLFW key callback -- press/repeat/release -- which is what "hold W to keep
walking" needs. Each rendered frame we read the held keys, turn them into a desired
(speed, heading) relative to the camera, advance the motion matcher one frame, write the
resulting qpos into MjData and draw. A follow-camera keeps the character centred.

Controls
  W / A / S / D ........ move (forward / left / back / right), relative to the camera
  Arrow keys ........... face direction, independent of travel (GenoView-style)
  Shift (hold) ......... walk instead of run (full stick is run pace, GenoView-style)
  B .................... box action: pick up when near the box, set down while carrying
  J .................... jump (transitions into a jump clip's run-up, then rides it)
  Space ................ reset to the start pose
  T .................... toggle the command trajectory gizmo (GenoView-style)
  Left-drag ............ orbit camera     Right-drag ... pan     Scroll ... zoom
  Esc .................. quit
"""
import math
import numpy as np
import glfw
import mujoco

from . import config as C

# GenoView draws the command trajectory in red: a sphere at each predicted future
# position plus a short stick pointing in the predicted facing direction.
_TRAJ_RGBA = np.array([0.9, 0.1, 0.1, 1.0], np.float32)
_TRAJ_Z = 0.05          # draw the ground gizmo just above the floor
_SPHERE_R = 0.05        # GenoView DrawSphere radius
_STICK_LEN = 0.25       # GenoView facing-stick length
_STICK_W = 0.012        # facing-stick / connector radius


# Movement keys (WASD = travel) and facing keys (arrows = independent facing) -> held set.
_MOVE_KEYS = {glfw.KEY_W, glfw.KEY_A, glfw.KEY_S, glfw.KEY_D}
_FACE_KEYS = {glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT}


class InteractiveViewer:
    def __init__(self, model, data, matcher, width=1280, height=720,
                 title="Motion Matching G1 - WASD to move, Shift to run"):
        self.model = model
        self.data = data
        self.matcher = matcher

        if not glfw.init():
            raise RuntimeError(
                "glfw.init() failed -- this viewer needs a display. On a headless host, "
                "run on a machine with a GPU/X display (MUJOCO_GL=glfw)."
            )
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create a GLFW window (no display available?).")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)   # vsync

        # MuJoCo visualization objects.
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # Follow-camera: orbit around the character at a comfortable height.
        self.cam.azimuth = 135.0
        self.cam.elevation = -20.0
        self.cam.distance = 4.0
        self.cam.lookat[:] = [0.0, 0.0, 0.8]

        # Input state.
        self.held = set()
        self.shift = False
        self.show_traj = True            # draw the command trajectory gizmo (toggle: T)
        self._speed = 0.0                # latest command speed, kept for the HUD
        self._mouse_last = None
        self._button = {"left": False, "right": False}

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)

        # If the scene carries the interactive box (qpos extends past the robot's 36), seed it
        # at the matcher's spawn so it is visible before the first simulation step.
        self.has_box = self.model.nq >= 43
        if self.has_box:
            self.data.qpos[36:43] = self.matcher.box_qpos()

    # --- input callbacks -----------------------------------------------------
    def _on_key(self, window, key, scancode, action, mods):
        self.shift = bool(mods & glfw.MOD_SHIFT)
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_SPACE:
                self.matcher.reset()
            elif key == glfw.KEY_T:
                self.show_traj = not self.show_traj
            elif key == glfw.KEY_J:
                self.matcher.trigger_jump()
            elif key == glfw.KEY_B:
                self.matcher.trigger_box()
            elif key in _MOVE_KEYS or key in _FACE_KEYS:
                self.held.add(key)
        elif action == glfw.RELEASE:
            self.held.discard(key)

    def _on_mouse_button(self, window, button, action, mods):
        press = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._button["left"] = press
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._button["right"] = press
        self._mouse_last = glfw.get_cursor_pos(window) if press else None

    def _on_cursor(self, window, xpos, ypos):
        if self._mouse_last is None:
            return
        dx = xpos - self._mouse_last[0]
        dy = ypos - self._mouse_last[1]
        self._mouse_last = (xpos, ypos)
        w, h = glfw.get_window_size(window)
        if self._button["left"]:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif self._button["right"]:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            return
        mujoco.mjv_moveCamera(self.model, action, dx / h, dy / h, self.scene, self.cam)

    def _on_scroll(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                              0.0, -0.05 * yoffset, self.scene, self.cam)

    # --- per-frame command from the keys -------------------------------------
    def _command(self):
        """Camera-relative WASD -> desired velocity; arrow keys -> an independent facing
        direction (genoview's left/right stick). Returns (desiredVel[3] m/s, desiredFace[3]
        unit; facing is zero when no arrow is held, so the controller faces the travel dir)."""
        fwd = math.radians(self.cam.azimuth + 180.0)   # ground heading "into the screen"
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

        m = np.linalg.norm(move)
        if m > 1e-6:   # full stick = MAX_SPEED; while carrying, the slower CARRY cap (the
            top = (C.CARRY_MAX_SPEED if self.matcher.state == C.SKILL_CARRY  # box data is slow)
                   else C.MAX_SPEED)
            move = move / m * (top * (C.WALK_SCALE if self.shift else 1.0))
        else:
            move = np.zeros(3)
        f = np.linalg.norm(face)
        face = face / f if f > 1e-6 else np.zeros(3)
        return move, face

    # --- main loop -----------------------------------------------------------
    def run(self):
        last = glfw.get_time()
        acc = 0.0
        speed = 0.0
        while not glfw.window_should_close(self.window):
            now = glfw.get_time()
            acc += now - last
            last = now

            # Advance the matcher at a fixed 30 Hz, independent of render rate.
            while acc >= C.DT:
                vel, face = self._command()
                self._speed = float(np.linalg.norm(vel))
                world = self.matcher.step(vel, face)
                self.data.qpos[0:36] = world
                if self.has_box:                       # box freejoint rides at qpos[36:43]
                    self.data.qpos[36:43] = self.matcher.box_qpos()
                mujoco.mj_forward(self.model, self.data)
                acc -= C.DT

            # Follow-camera tracks the character root.
            self.cam.lookat[0] = float(self.data.qpos[0])
            self.cam.lookat[1] = float(self.data.qpos[1])

            w, h = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, w, h)
            mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                   mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            if self.show_traj:
                self._draw_command()
            mujoco.mjr_render(viewport, self.scene, self.ctx)
            self._overlay(viewport, self._speed)

            glfw.swap_buffers(self.window)
            glfw.poll_events()
        glfw.terminate()

    # --- command trajectory gizmo (GenoView DrawTrajectory) ------------------
    def _draw_command(self):
        """Append the spring-predicted command trajectory (matcher.Tpos / Tdir) to the
        scene: a red sphere at each future tap with a short stick along its facing."""
        for (px, py, _), (dx, dy, _) in zip(self.matcher.Tpos, self.matcher.Tdir):
            base = np.array([px, py, _TRAJ_Z])
            self._add_sphere(base, _SPHERE_R)
            self._add_stick(base, base + _STICK_LEN * np.array([dx, dy, 0.0]))

    def _next_geom(self):
        if self.scene.ngeom >= self.scene.maxgeom:
            return None
        g = self.scene.geoms[self.scene.ngeom]
        self.scene.ngeom += 1
        return g

    def _add_sphere(self, pos, radius):
        g = self._next_geom()
        if g is None:
            return
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([radius, 0.0, 0.0]), np.asarray(pos, float),
                            np.eye(3).flatten(), _TRAJ_RGBA)

    def _add_stick(self, p0, p1):
        g = self._next_geom()
        if g is None:
            return
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                            np.zeros(3), np.zeros(3), np.eye(3).flatten(), _TRAJ_RGBA)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, _STICK_W,
                             np.asarray(p0, float), np.asarray(p1, float))

    def _overlay(self, viewport, speed):
        m = self.matcher
        # Box state machine takes precedence in the HUD; otherwise show the loco gait.
        state = m.state_name()
        if state == "LOCOMOTION" and not m.jumping:
            head = ("RUN" if speed > C.MAX_SPEED * (1 + C.WALK_SCALE) / 2 else
                    ("WALK" if speed > 1e-3 else "IDLE"))
            if self.has_box:
                head += "  [B: pick up]" if m.near_box else "  (walk to the box, then B)"
        else:
            head = "JUMP" if m.jumping else state
            if state == "CARRY":
                head += "  [B: set down]"
        lib, cur = m.lib, m.cur
        cid = int(lib["clip_id"][cur])
        clip = lib["clip_names"][cid]
        fic, length = int(lib["frame_in_clip"][cur]), int(lib["lengths"][cid])
        title = f"{head}   {speed:.1f} m/s"
        body = (f"clip [{cid}]: {clip}\n"
                f"frame: {fic}/{length - 1}  (global {cur})\n"
                f"box: {'held' if m.box_held else 'resting'}"
                f"   command gizmo: {'on' if self.show_traj else 'off'} (T)\n"
                "WASD move | arrows face | Shift walk | B box | J jump | Space reset\n"
                "drag orbit | right-drag pan | scroll zoom | Esc quit")
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                           mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport,
                           title, body, self.ctx)
