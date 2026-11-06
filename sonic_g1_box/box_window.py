"""UI 2: run_slam.py's window. The loco driver's window (motionmatching-g1-loco sonic_g1_loco/mm_driver.py
run_window: its keys, the measured robot at its SLAM pelvis, the reference it plays, the map around the robot, the
lab's furniture, the command, UI 1's trajectory) with the box task on top: the controller's box, the map server's
reports, the box keys, the box rows of the status panel and a legend."""
import math
import os
import time

import glfw
import mujoco
import numpy as np

import operator_link as OP
from deploy_link import MODE_IDLE, MODE_PLANNER, MODE_POSE, MODE_STOPPED
from mm_g1.mathq import qfrom_yaw, qmul, qyaw

SCENE_XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'unitree_g1', 'scene.xml')
DRAW_S = 1.0 / 60               # s between drawn frames: the bridge ticks every few ms in between
POSE_POLL_S = 1.0 / 30          # s between the window's pose requests, for a smooth robot (the bridge asks at 10 Hz)
BOX_RGBA = np.array([0.82, 0.52, 0.22, 1.0], np.float32)    # the box's brown, UI 1's too
HELD_RGBA = np.array([0.25, 0.9, 0.35, 1.0], np.float32)
PLACE_RGBA = np.array([1.0, 0.85, 0.2, 0.45], np.float32)
CLICK_PX = 5                    # px the cursor may move between press and release for a click, not a camera drag
CLICK_NEAR_M = 0.4              # m: a click that misses every box takes the one this near where its ray meets its height
OUTLINE_R, OUTLINE_R_PICK = 0.006, 0.013   # m: edge radius of a reported box's outline, the controller's or x's one
OUTLINE_EMISSION = 0.25         # the thin edges read dark under the scene's light otherwise
BOX_EDGES = ((0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7))
ROBOT_LABEL_DZ, REF_LABEL_DZ = 0.62, 0.82   # m above torso_link: the two labels stack when the figures overlap (POSE)
BOX_LABEL_DZ, REPORT_LABEL_DZ = 0.10, 0.30  # m above a box's top: the linked report and the controller's box overlap
PATH_EVERY = 5                  # frames between the dots of the reference's path ahead
HEIGHT_CAP = 1.0                # m: the map's heights are drawn up to this (the loco driver's scandot clip)
OVERLAY_CHARS = 499             # mjr_overlay's buffer per column (MuJoCo 3.3), measured
GHOST_WORD = {MODE_IDLE: 'gray', MODE_PLANNER: 'yellow', MODE_POSE: 'blue', MODE_STOPPED: 'dark gray'}
GHOST_RGBA = {MODE_IDLE: np.array([0.6, 0.6, 0.6, 0.6], np.float32),                  # lmm-g1-box's yellow and its ghost's blue
              MODE_PLANNER: np.array([0.99, 0.77, 0.10, 0.7], np.float32),
              MODE_POSE: np.array([0.30, 0.67, 0.97, 0.7], np.float32), MODE_STOPPED: np.array([0.45, 0.45, 0.45, 0.5], np.float32)}
MODE_WORD = {MODE_IDLE: 'IDLE', MODE_PLANNER: 'PLANNER', MODE_POSE: 'POSE', MODE_STOPPED: 'STOPPED'}
KEYS_L = ']\nP\nO\nEnter\nW A S D\nArrows\nShift\nF\nT\nEsc\n\nX\nClick\nB\nC\nV\nL\nMouse'
KEYS_R = ('start control (planner mode)\ntoggle planner <-> POSE\nstop -> damping (final)\n'
          're-send mode\nmove\nface\nhold to run\nWASD frame heading/camera\n'
          'drive the UI 1 trajectory on/off\nquit\n'
          '\npick up the box reported nearest the robot (again: the next one)\n'
          'pick up the clicked box (in locomotion only)\nput the held box down ahead\n'
          'cancel the pick or the place\ntoggle the path ahead\ntoggle the legend (what each shape is)\n'
          'drag orbits, right-drag pans, scroll zooms')
COMMAND_RGBA = (0.89, 0.34, 0.29, 1.0)


def keyboard_stick(window, stick, walk_scale=0.4, cam_azimuth=None):
    """A copy of the loco repo's genoview_g1.keyboard_stick: a world-plane (3,) vector, 'left' = WASD move,
    'right' = arrows face, left Shift scales it to walk_scale. With cam_azimuth (deg) W/Up is along that heading and
    A/Left to its left, else W is world +x."""
    def down(k):
        return glfw.get_key(window, k) == glfw.PRESS

    x = y = 0.0
    if stick == 'left':
        if down(glfw.KEY_W): x += 1.0
        if down(glfw.KEY_S): x -= 1.0
        if down(glfw.KEY_A): y += 1.0
        if down(glfw.KEY_D): y -= 1.0
    else:
        if down(glfw.KEY_UP): x += 1.0
        if down(glfw.KEY_DOWN): x -= 1.0
        if down(glfw.KEY_LEFT): y += 1.0
        if down(glfw.KEY_RIGHT): y -= 1.0
    mag = np.hypot(x, y)
    if mag > 0.0:
        speed = walk_scale if down(glfw.KEY_LEFT_SHIFT) else 1.0
        x, y = x / mag * speed, y / mag * speed
    if cam_azimuth is not None:
        a = np.radians(cam_azimuth)
        v = x * np.array([np.cos(a), np.sin(a)]) + y * np.array([-np.sin(a), np.cos(a)])
        return np.array([v[0], v[1], 0.0])
    return np.array([x, y, 0.0])


def is_active(o):
    return str(o.get('track_state', 'ACTIVE')).upper() != 'INACTIVE'


def ray_hits_box(orig, dirn, o):
    """Distance along the ray (orig, dirn: (3,), dirn unit) to the reported object's box, or None."""
    T = np.asarray(o['T_world_object'], float).reshape(4, 4)
    R, c = T[:3, :3], np.asarray(o['bbox_center_world'], float)
    lo_, ld = R.T @ (orig - c), R.T @ dirn                      # (3,) in the box frame
    half = 0.5 * np.asarray(o['bbox_size_xyz'], float)
    t0, t1 = 0.0, np.inf
    for k in range(3):
        if abs(ld[k]) < 1e-9:
            if abs(lo_[k]) > half[k]:
                return None
            continue
        a, b = sorted(((-half[k] - lo_[k]) / ld[k], (half[k] - lo_[k]) / ld[k]))
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return None
    return float(t0)


def fit_overlay(rows):
    """The two columns of an mjr_overlay panel from (label, value) rows. mjr_overlay keeps OVERLAY_CHARS of each column
    and past that the columns no longer line up, so the longest values are cut first."""
    vals = [r[1] for r in rows]
    while (over := len('\n'.join(vals)) - OVERLAY_CHARS) > 0:
        i = max(range(len(vals)), key=lambda k: len(vals[k]))
        vals[i] = vals[i][:max(0, len(vals[i]) - over - 2)].rstrip() + '..'
    return '\n'.join(r[0] for r in rows), '\n'.join(vals)


def quat_mat(q):
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(q, float))
    return mat


def run_window(br, args):
    br.map.retry_s = 1.0      # no map server: each unanswered request would block the window for its timeout
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    half = 0.5 * np.asarray(args.box_size, float)                # (3,)

    class Window:
        def __init__(self):
            if not glfw.init():
                raise RuntimeError('glfw.init() failed: the window needs a display (--terminal runs without one)')
            self.window = glfw.create_window(1400, 900, 'G1 box bridge (SLAM) -- ] start, P mode, X pick up, B put down, '
                                             'O stop', None, None)
            if not self.window:
                glfw.terminate()
                raise RuntimeError('no GLFW window (no display?): --terminal runs without one')
            glfw.make_context_current(self.window)
            glfw.swap_interval(0)               # the loop paces itself: a blocking swap would slow the bridge's ticks
            self.cam, self.opt = mujoco.MjvCamera(), mujoco.MjvOption()
            self.scene = mujoco.MjvScene(model, maxgeom=10000)
            self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
            self.cam.azimuth, self.cam.elevation, self.cam.distance = 150.0, -25.0, 4.5
            self.cam.lookat[:] = (0.0, 0.0, 0.8)
            self.drag = {'last': None, 'left': False, 'right': False}
            glfw.set_key_callback(self.window, self._on_key)
            glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
            glfw.set_cursor_pos_callback(self.window, self._on_cursor)
            glfw.set_scroll_callback(self.window, self._on_scroll)
            self.gdata = mujoco.MjData(model)    # the stick figure: FK only
            self.ghost_bodies = list(range(1, model.nbody))
            feet = {model.body(n).id for n in ('left_ankle_roll_link', 'right_ankle_roll_link')}
            self.foot_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in feet]
            self.corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)   # (8, 3)
            self.torso = model.body('torso_link').id
            self.pose_t = 0.0
            self.robot_why = 'starting'
            self.follow = {'xy': None, 't': 0.0}
            self.click = None
            self.show_legend = True
            self.show_path = True
            self.op = OP.Operator(br.map, 'box')
            self.auto = False
            self.obstacles = {'boxes': [], 'pad': [], 'dynamic': [], 't': 0.0}
            self.shapes = {'list': [], 't': 0.0}
            self.quit = False
            # the loco driver's keys, then the box task's on keys it leaves free (Q quits no more: it is next to W, A)
            self.actions = {glfw.KEY_RIGHT_BRACKET: br.start, glfw.KEY_P: br.toggle_pose, glfw.KEY_O: br.stop,
                            glfw.KEY_ENTER: br.resend_mode, glfw.KEY_F: self._toggle_frame, glfw.KEY_T: self._toggle_drive,
                            glfw.KEY_ESCAPE: self._quit,
                            glfw.KEY_X: br.pick_nearest, glfw.KEY_B: br.place_ahead, glfw.KEY_C: br.cancel,
                            glfw.KEY_V: self._toggle_path, glfw.KEY_L: self._toggle_legend}

        def _on_key(self, window, key, scancode, action, mods):
            if action == glfw.PRESS and key in self.actions:
                self.actions[key]()

        def _on_mouse_button(self, window, button, action, mods):
            if button == glfw.MOUSE_BUTTON_LEFT:
                if action == glfw.PRESS:
                    self.click = glfw.get_cursor_pos(window)
                elif action == glfw.RELEASE and self.click is not None:
                    x, y = glfw.get_cursor_pos(window)
                    if math.hypot(x - self.click[0], y - self.click[1]) <= CLICK_PX:
                        self._click_select(x, y)
                    self.click = None
            press = action == glfw.PRESS
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.drag['left'] = press
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.drag['right'] = press
            self.drag['last'] = glfw.get_cursor_pos(window) if press else None

        def _on_cursor(self, window, x, y):
            if self.drag['last'] is None:
                return
            dx, dy = x - self.drag['last'][0], y - self.drag['last'][1]
            self.drag['last'] = (x, y)
            if self.drag['left']:
                how = mujoco.mjtMouse.mjMOUSE_ROTATE_V
            elif self.drag['right']:
                how = mujoco.mjtMouse.mjMOUSE_MOVE_V
            else:
                return
            h = glfw.get_window_size(window)[1]
            mujoco.mjv_moveCamera(model, how, dx / h, dy / h, self.scene, self.cam)

        def _on_scroll(self, window, xoffset, yoffset):
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, self.scene, self.cam)

        def _quit(self):
            self.quit = True

        def _toggle_legend(self):
            self.show_legend = not self.show_legend

        def _toggle_path(self):
            self.show_path = not self.show_path

        def _toggle_frame(self):
            args.frame = 'camera' if args.frame == 'heading' else 'heading'
            if br.flags is not None:
                br.flags.note_live('frame', args.frame)

        def _toggle_drive(self):
            rf = br.route_follow
            rf.set_drive(not rf.drive)
            br.say(f'drive the desired trajectory: {"on" if rf.drive else "off"}')

        def _sticks(self, now):
            """The loco driver's: WASD and the arrows, then UI 3's held keys, else UI 1's trajectory while Drive is
            on. A pick or a place (X, B, the grip check) goes before the trajectory. UI 3's commands too."""
            for c in self.op.poll(now):
                br.operator_command(c, lambda: (br.close(), glfw.terminate()))
            az = math.degrees(br.heading()) if args.frame == 'heading' else self.cam.azimuth
            left = keyboard_stick(self.window, 'left', args.run_scale, az)
            right = keyboard_stick(self.window, 'right', 1.0, az)
            self.auto = br.drive_sticks(now, self.op, (left, right))

        def _click_select(self, x, y):
            """The reported box under the cursor: the first one its ray enters, else the one
            nearest where the ray comes down to its centre's height."""
            objs = [o for o in br.reports.objects if o.get('bbox_center_world')]
            if not objs:
                return                                   # a click on empty space (or to focus the window)
            w, h = glfw.get_window_size(self.window)
            selpnt = np.zeros(3)
            gid, fid, sid = (np.zeros((1, 1), np.int32) for _ in range(3))
            mujoco.mjv_select(model, data, self.opt, w / h, x / w, (h - y) / h, self.scene, selpnt, gid, fid, sid)
            if gid[0, 0] < 0 and fid[0, 0] < 0 and sid[0, 0] < 0:
                return                                   # the sky: no ray direction from mjv_select
            orig = 0.5 * (np.asarray(self.scene.camera[0].pos, float) + np.asarray(self.scene.camera[1].pos, float))
            dirn = selpnt - orig
            dirn /= np.linalg.norm(dirn) + 1e-12
            hits = [(t, o) for o in objs if (t := ray_hits_box(orig, dirn, o)) is not None]
            if hits:
                o = min(hits, key=lambda x: x[0])[1]
            else:
                best = None
                for o in objs:
                    c = np.asarray(o['bbox_center_world'], float)
                    if abs(dirn[2]) < 1e-6:
                        continue
                    p = orig + dirn * (c[2] - orig[2]) / dirn[2]
                    d = float(np.hypot(*(p[:2] - c[:2])))
                    if d < CLICK_NEAR_M and (best is None or d < best[0]):
                        best = (d, o)
                if best is None:
                    return
                o = best[1]
            br.select_report(o)

        def _geom(self, kind, size, pos, rgba, mat=None):
            if self.scene.ngeom >= self.scene.maxgeom:
                return None
            g = self.scene.geoms[self.scene.ngeom]
            mujoco.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float),
                                np.eye(3).ravel() if mat is None else np.asarray(mat, float), np.asarray(rgba, np.float32))
            self.scene.ngeom += 1
            return g

        def _label(self, pos, text):
            """White text with a shadow at pos (3,), drawn by mjr_render whatever the geom's colour."""
            g = self._geom(mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3), pos, (1.0, 1.0, 1.0, 1.0))
            if g is not None:
                g.label = text[:99]

        def _edge(self, a, b, radius, rgba):
            g = self._geom(mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), rgba)
            if g is not None:
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, np.asarray(a, float), np.asarray(b, float))
                g.emission = OUTLINE_EMISSION

        def _outline(self, o, radius, rgba):
            """A reported box as its 12 edges: an outline, unlike the controller's solid box."""
            T = np.asarray(o['T_world_object'], float).reshape(4, 4)
            h = 0.5 * np.asarray(o['bbox_size_xyz'], float)
            c = np.asarray(o['bbox_center_world'], float)
            pts = c + (self.corners * h) @ T[:3, :3].T                       # (8, 3) map
            for i, j in BOX_EDGES:
                self._edge(pts[i], pts[j], radius, rgba)
            return c + np.array([0.0, 0.0, h[2] + REPORT_LABEL_DZ])

        def _yaw_mat(self, yaw):
            c, s = math.cos(yaw), math.sin(yaw)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float).ravel()

        def _draw_obstacles(self):
            """The real map around the robot, as the loco driver's window and UI 1 draw it: orange to dark red by
            height, the padding as flat light-pink tiles, tracked blobs not in the static map gray (new) or orange
            (moving, with the disc swept 0.5 s along their velocity)."""
            ob = self.obstacles
            if time.monotonic() - ob['t'] > 0.5:
                ob['t'] = time.monotonic()
                r = br.map.request({'op': 'obstacles', 'radius': args.obstacle_radius})
                if r is not None:
                    ob['boxes'], ob['pad'], ob['dynamic'] = r['boxes'], r['pad'], r.get('dynamic', [])
            G = mujoco.mjtGeom
            for d in ob['dynamic']:
                if d['confirmed']:
                    continue
                h = max(d['h'], 0.05)
                rgba = (0.88, 0.35, 0.16, 0.55) if d['moving'] else (0.5, 0.5, 0.5, 0.35)
                self._geom(G.mjGEOM_CYLINDER, (d['r'], 0.5 * h, 0), (d['x'], d['y'], 0.5 * h), rgba)
                if d['moving']:
                    self._geom(G.mjGEOM_CYLINDER, (d['r'], 0.5 * h, 0),
                               (d['x'] + 0.5 * d['vx'], d['y'] + 0.5 * d['vy'], 0.5 * h), (0.88, 0.35, 0.16, 0.2))
            for x, y, hx, hy in ob['pad']:
                self._geom(G.mjGEOM_BOX, (hx, hy, 0.004), (x, y, 0.004), (0.97, 0.77, 0.75, 0.55))
            for x, y, hx, hy, top in ob['boxes']:
                t = min(top / HEIGHT_CAP, 1.0)
                self._geom(G.mjGEOM_BOX, (hx, hy, 0.5 * top), (x, y, 0.5 * top),
                           (1.0 - 0.24 * t, 0.96 - 0.84 * t, 0.9 - 0.78 * t, 0.7))

        def _draw_shapes(self):
            """The map server's virtual objects and the lab's furniture (one box per link footprint), purple."""
            sh_ = self.shapes
            if time.monotonic() - sh_['t'] > 1.0:
                sh_['t'] = time.monotonic()
                r = br.map.request({'op': 'shapes'})
                if r is not None:
                    sh_['list'] = r['shapes']
            G, rgba = mujoco.mjtGeom, (0.49, 0.30, 0.84, 0.45)
            for sh in sh_['list']:
                if sh['kind'] == 'part':
                    for pl in sh.get('polys', []):
                        q = np.asarray(pl['poly'], float)                                 # (4, 2) corners
                        a, b = q[1] - q[0], q[3] - q[0]
                        top = max(min(pl['top'], HEIGHT_CAP), 0.02)
                        self._geom(G.mjGEOM_BOX, (0.5 * np.linalg.norm(a), 0.5 * np.linalg.norm(b), 0.5 * top),
                                   (*q.mean(0), 0.5 * top), rgba, self._yaw_mat(math.atan2(a[1], a[0])))
                    continue
                h = max(sh['h'], 0.02)
                if sh['kind'] == 'circle':
                    self._geom(G.mjGEOM_CYLINDER, (sh['r'], 0.5 * h, 0), (sh['x'], sh['y'], 0.5 * h), rgba)
                else:
                    self._geom(G.mjGEOM_BOX, (0.5 * sh['sx'], 0.5 * sh['sy'], 0.5 * h), (sh['x'], sh['y'], 0.5 * h),
                               rgba, self._yaw_mat(sh.get('yaw', 0.0)))

        def _draw_command(self):
            """br.command_local() at the robot's SLAM pose, as UI 3 draws it too."""
            if br.slam is None:
                return
            local, yaw = br.command_local(), float(br.slam['root_yaw'])   # (3, 4)
            R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            G = mujoco.mjtGeom
            for p, u in zip(np.asarray(br.slam['root_xy'][:2], float) + local[:, :2] @ R.T, local[:, 2:] @ R.T):
                base = np.array([p[0], p[1], 0.05])
                self._geom(G.mjGEOM_SPHERE, (0.042, 0, 0), base, COMMAND_RGBA)
                gm = self._geom(G.mjGEOM_ARROW, np.zeros(3), np.zeros(3), COMMAND_RGBA)
                if gm is not None:
                    mujoco.mjv_connector(gm, G.mjGEOM_ARROW, 0.015, base, base + 0.32 * np.array([u[0], u[1], 0.0]))

        def _draw_route(self):
            """UI 1's trajectory as the follower holds it: the target waypoint larger, an arrow from the robot to it
            while the follower steers."""
            rf = br.route_follow
            f = rf.follower
            if f is None:
                return
            on = rf.drive and not f.done and self.auto
            rgba = (0.1, 0.55, 0.65, 0.9) if rf.drive and not f.done else (0.4, 0.5, 0.55, 0.5)
            G = mujoco.mjtGeom
            pts = [(x, y, 0.03) for x, y in f.wp]
            for i, p in enumerate(pts):
                self._geom(G.mjGEOM_SPHERE, (0.07 if on and i == f.idx else 0.035, 0, 0), p, rgba)
            for a, b in zip(pts, pts[1:] + (pts[:1] if f.loop else [])):
                gm = self._geom(G.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), rgba)
                if gm is not None:
                    mujoco.mjv_connector(gm, G.mjGEOM_CAPSULE, 0.012, np.asarray(a, float), np.asarray(b, float))
            if on and br.slam is not None and f.idx < len(pts):
                gm = self._geom(G.mjGEOM_ARROW, np.zeros(3), np.zeros(3), rgba)
                if gm is not None:
                    mujoco.mjv_connector(gm, G.mjGEOM_ARROW, 0.02, np.array([*br.slam['root_xy'][:2], 0.06]),
                                         np.asarray(pts[f.idx], float) + (0, 0, 0.03))

        def _pose_robot(self):
            """The measured robot at its SLAM pelvis pose, never at the reference's (that one runs ahead of the robot).
            Hidden without g1_debug or a pose."""
            now = time.monotonic()
            if now - self.pose_t >= POSE_POLL_S:
                self.pose_t = now
                br.poll_pose()
            data.qpos[0:7] = (0.0, 0.0, -5.0, 1.0, 0.0, 0.0, 0.0)
            state = br.robot.get()
            if state is None or br.slam is None:
                self.robot_why = ('no g1_debug from the deploy node (5557)' if state is None else
                                  'no SLAM pose from the map server')
                return False
            self.robot_why = None
            bq, q = state
            data.qpos[3:7] = qmul(qfrom_yaw(float(br.slam['root_yaw']) - qyaw(list(bq))), list(bq))
            data.qpos[7:36] = q
            data.qpos[0:3] = (float(br.slam['root_xy'][0]), float(br.slam['root_xy'][1]), 1.0)
            mujoco.mj_kinematics(model, data)
            low = min(float((data.geom_xpos[g] + (model.geom_aabb[g, :3] + self.corners * model.geom_aabb[g, 3:])
                             @ data.geom_xmat[g].reshape(3, 3).T)[:, 2].min()) for g in self.foot_geoms)
            data.qpos[2] -= low - 0.002
            return True

        def _report_roles(self, objs):
            """(the report linked to the controller's box or None, the one x takes now or None)."""
            linked = next((o for o in objs if br.box_id is not None and int(o['track_id']) == br.reports.track), None)
            order = br.ordered_reports()
            nxt = br.next_report(order) if order else None
            return linked, nxt

        def _draw_reports(self, robot_xy):
            objs = [o for o in br.reports.objects if o.get('bbox_center_world')]
            linked, nxt = self._report_roles(objs)
            for o in objs:
                lost = not is_active(o)
                rgba = [*BOX_RGBA[:3], 0.35 if lost else 1.0]
                top = self._outline(o, OUTLINE_R_PICK if o is linked or (o is nxt and linked is None) else OUTLINE_R, rgba)
                dist = float(np.hypot(*(np.asarray(o['bbox_center_world'][:2], float) - robot_xy[:2])))
                role = ("the controller's box" if o is linked else
                        ('x takes this' if linked is None else 'x: the next one') if o is nxt else '')
                text = f"report #{o.get('track_id', '?')} {o.get('language_label') or 'object'}, {dist:.1f} m"
                self._label(top, text + (f', {role}' if role else '') + (', out of view' if lost else ''))

        def _box_state(self, frame, b):
            w = br.sim.world
            if frame.held == b['id']:
                return 'held'
            if w.pick_id == b['id']:
                return 'regrasp' if w.regrasp == b['id'] else 'pick'
            return 'floor'

        def _draw_boxes(self, frame):
            """The controller's boxes: on the floor where the reports put them, or in the reference's hands while the
            played frame holds one."""
            drawn = set()
            for b in br.sim.world.boxes:
                if frame.held == b['id'] and frame.ghost_box is not None:
                    pos, quat_, rgba = frame.ghost_box[0:3], frame.ghost_box[3:7], HELD_RGBA
                else:
                    pos, quat_, rgba = (b['pos'][0], b['pos'][1], half[2]), qfrom_yaw(b['yaw']), BOX_RGBA
                self._geom(mujoco.mjtGeom.mjGEOM_BOX, half, pos, rgba, quat_mat(quat_))
                word = {'held': 'held (reference)', 'pick': 'to pick up', 'regrasp': 'to pick up again (grip check)',
                        'floor': 'on the floor'}[self._box_state(frame, b)]
                self._label(np.asarray(pos, float) + np.array([0.0, 0.0, half[2] + BOX_LABEL_DZ]),
                            f"controller's box {b['label']}: {word}")
                drawn.add(b['id'])
            if frame.held is not None and frame.held not in drawn and frame.ghost_box is not None:
                self._geom(mujoco.mjtGeom.mjGEOM_BOX, half, frame.ghost_box[0:3], HELD_RGBA, quat_mat(frame.ghost_box[3:7]))
            t = br.sim.world.place_target
            if t is not None:
                self._geom(mujoco.mjtGeom.mjGEOM_BOX, [half[0], half[1], 0.004], (t['pos'][0], t['pos'][1], 0.004),
                           PLACE_RGBA, quat_mat(qfrom_yaw(t['yaw'])))
                self._label((t['pos'][0], t['pos'][1], 0.25), 'place target (b)')

        def _draw_reference(self, frame):
            g = self.gdata
            g.qpos[0:36] = frame.qpos[0:36]
            mujoco.mj_kinematics(model, g)
            rgba = GHOST_RGBA[br.mode]
            for b in self.ghost_bodies:
                pa = model.body_parentid[b]
                a, c = g.xpos[pa], g.xpos[b]
                if pa == 0 or np.linalg.norm(c - a) < 1e-6:
                    continue
                gm = self._geom(mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), rgba)
                if gm is not None:
                    mujoco.mjv_connector(gm, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.025, np.asarray(a, float), np.asarray(c, float))
            self._label(g.xpos[self.torso] + np.array([0.0, 0.0, REF_LABEL_DZ]), f'reference ({MODE_WORD[br.mode]})')
            if self.show_path:
                r = br.runner
                dot = np.array([*rgba[:3], 0.8], np.float32)
                for f in range(br.cur + PATH_EVERY, r.produced, PATH_EVERY):
                    q = r.frame_at(f).qpos
                    self._geom(mujoco.mjtGeom.mjGEOM_SPHERE, (0.025, 0.0, 0.0), (q[0], q[1], 0.03), dot)

        def _legend(self, frame, live):
            """(what you see, what it is) for the top-right panel, with the current state."""
            if live:
                robot = 'measured robot (g1_debug) at its SLAM pelvis pose'
                age = time.time() - br.slam_wall
                if age > 1.0:
                    robot += f', pose {age:.0f} s old'
            else:
                robot = f'hidden: {self.robot_why}'
            if br.mode == MODE_POSE:
                ref = "moved onto the robot's SLAM pose every 0.2 s, not while bent down"
            elif br.mode == MODE_STOPPED:
                ref = 'stopped: the node no longer tracks it'
            else:
                ref = 'put on the robot every 0.2 s, the robot plays it from POSE (P)'
            rows = [('robot model', robot),
                    (f'{GHOST_WORD[br.mode]} stick figure', 'reference the robot plays: gray IDLE, yellow PLANNER, blue POSE'),
                    ('', ref)]
            if br.sim.world.boxes:
                rows += [('solid brown box', "controller's box, from a report: it walks there, picks it up"),
                         ('green box', "held in the reference's hands (not measured)")]
            else:
                rows.append(('no solid box', "x or a click puts a reported box in the controller's world"))
            rows += [('brown outlines', "boxes the map server reports (SAM3 or the sim's truth)"),
                     ('', "thick = the controller's box or x's next, dim = unseen for 1 s")]
            if br.sim.world.place_target is not None:
                rows.append(('yellow patch', 'where the box is put down (b)'))
            rows.append(('small balls', 'reference pelvis path ahead, 0.1 s apart (V hides)') if self.show_path else
                        ('path off', 'V shows the reference pelvis path ahead'))
            rows.append(('red balls, arrows', 'the command (keys, route): 0.33 to 1 s ahead'))
            rows.append(('orange to red blocks', 'the map around the robot, by height'))
            if self.shapes['list']:
                rows.append(('purple blocks', "the lab's furniture and UI 1's virtual objects"))
            if br.route_follow.follower is not None:
                rows.append(('teal balls, line', "UI 1's trajectory (T walks it)"))
            rows.append(('dark floor patches', 'shadows'))
            return fit_overlay(rows)

        def _status(self, frame, live):
            w, c, r = br.sim.world, br.sim.controller, br.runner
            state, hud, sem = br.task_words(frame)
            mode = {MODE_IDLE: 'IDLE       next: ] (planner mode, the robot stands)',
                    MODE_PLANNER: "PLANNER    the node's planner (WASD walks it); next: P -> POSE",
                    MODE_POSE: 'POSE       the controller drives the robot',
                    MODE_STOPPED: 'STOPPED    damping; restart deploy.sh to resume'}[br.mode]
            task = f"{hud}   [{state}, {c.mode} {c.semantics}]"
            b = br.world_box()
            if b is None:
                boxl = 'none: x takes the nearest report, a click the clicked one'
            else:
                last = br.reports.last
                seen = 'no report yet' if last is None else f"seen {time.time() - last['wall']:.1f} s ago"
                where = ('held by the reference' if frame.held == b['id'] else
                         f"at ({b['pos'][0]:+.2f}, {b['pos'][1]:+.2f}) m {math.degrees(b['yaw']):+.0f} deg")
                boxl = f"{b['label']} {where}, track #{br.reports.track}, {seen}"
            objs = [o for o in br.reports.objects if o.get('bbox_center_world')]
            n = len(objs)
            objl = f'{n} reported'
            if n:
                nxt = br.next_report(br.ordered_reports())
                d = math.hypot(*(np.asarray(nxt['bbox_center_world'][:2], float) - br.robot_xy()))
                objl += (f', x takes #{nxt["track_id"]}, {d:.2f} m away' if b is None else
                         ', x: the next one, click: that one' if n > 1 else '')
            if w.holding_id is not None:
                targ = ('put down at ({:+.2f}, {:+.2f}) m'.format(*w.place_target['pos']) if w.place_target else
                        'holding, b puts it down')
            elif w.pick_id is not None:
                targ = f"pick up box {w.find(w.pick_id)['label']}" + (' again (grip check)' if w.regrasp == w.pick_id else '')
            elif w.regrasp is not None:
                targ = 'the box left the hands: picked up again once the camera sees it'
            else:
                targ = 'none (x picks up, b puts down)'
            grip = (f'{br.drops} drop{"s" if br.drops != 1 else ""}' if br.drops else 'no drop')
            if w.holding_id is not None and br.mode == MODE_POSE:
                grip += f', {br.grip.count} of {args.grip_need} reports on the floor now'
                if br.grasp.gap is not None:
                    grip += f', palms {br.grasp.gap:.2f} m apart'
            g = br.grasp
            palm = ('off (--palm-force 0)' if g.force <= 0 else
                    f"{g.force:.0f} N {'pressing' if g.on else 'off'}, ramp {g.ramp:.0%}")
            if frame.open and not g.on:
                palm += f', shoulders open {g.open_rad:.1f} rad'
            p = br.slam
            locl = ('NOT LOCALIZED' if p is None else
                    f"({p['root_xy'][0]:+.2f}, {p['root_xy'][1]:+.2f}) {math.degrees(p['root_yaw']):+.0f} deg, "
                    f"{float(p.get('age_s', 0.0)) + time.time() - br.slam_wall:.2f} s old [{p.get('link') or '?'}]")
            ls = r.last_seed
            sync = ('-' if ls is None else f"{ls['d']:.2f} m {math.degrees(ls['dyaw']):+.1f} deg at the last seed")
            sync += (f', {r.seeds} seeds, {r.held_back} held back, {r.kept_out} kept out'
                     + (f', {br.seed_skips} jumps' if br.seed_skips else ''))
            robot = 'LIVE' if live else ('no g1_debug' if br.robot.get() is None else 'NOT LOCALIZED')
            rows = [('mode', mode), ('task', task), ('box', boxl), ('reports', objl), ('target', targ),
                    ('grip check', grip), ('box moves', f'{br.syncs} taken from the reports'), ('palm force', palm),
                    ('robot (SLAM)', locl), ('ref vs robot', sync), ('robot state', robot), ('link', br.link.endpoint),
                    ('trajectory', br.route_follow.text(self.auto))]
            return fit_overlay(rows)

        def draw(self):
            frame = br.frame_now()
            live = self._pose_robot()
            mujoco.mj_forward(model, data)
            # follow the robot (the reference without it), eased so SLAM jitter does not shake the view
            target = np.asarray(br.slam['root_xy'][:2], float) if live else np.asarray(frame.qpos[0:2], float)
            t = time.monotonic()
            if self.follow['xy'] is None or args.follow_s <= 0:
                self.follow['xy'] = target.copy()
            else:
                self.follow['xy'] += (1.0 - math.exp(-(t - self.follow['t']) / args.follow_s)) * (target - self.follow['xy'])
            self.follow['t'] = t
            self.cam.lookat[0], self.cam.lookat[1] = float(self.follow['xy'][0]), float(self.follow['xy'][1])
            fw, fh = glfw.get_framebuffer_size(self.window)
            vp = mujoco.MjrRect(0, 0, fw, fh)
            mujoco.mjv_updateScene(model, data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            self._draw_obstacles()
            self._draw_shapes()
            self._draw_command()
            self._draw_route()
            self._draw_reports(target)
            self._draw_boxes(frame)
            self._draw_reference(frame)
            if live:
                self._label(data.xpos[self.torso] + np.array([0.0, 0.0, ROBOT_LABEL_DZ]), 'robot (measured)')
            mujoco.mjr_render(vp, self.scene, self.ctx)
            labels, values = self._status(frame, live)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, vp, labels, values, self.ctx)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, vp, KEYS_L, KEYS_R,
                               self.ctx)
            if self.show_legend:
                looks, meanings = self._legend(frame, live)
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, vp, looks, meanings,
                                   self.ctx)
            if br.lost:
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOP, vp, 'BOX LOST: standing still',
                                   None, self.ctx)
            glfw.swap_buffers(self.window)

        def run(self, max_frames=None):
            frames, next_draw = 0, 0.0
            try:
                while not glfw.window_should_close(self.window) and not self.quit:
                    if max_frames is not None and frames >= max_frames:
                        break
                    now = time.monotonic()
                    self._sticks(now)
                    br.phase = br.sim.controller.semantics
                    br.tick(now)
                    if now >= next_draw:
                        next_draw = now + DRAW_S
                        frames += 1
                        self.draw()
                    glfw.poll_events()
                    time.sleep(0.003)
            except KeyboardInterrupt:
                pass
            finally:
                br.close()
                glfw.terminate()

    Window().run(max_frames=args.smoke_frames)
