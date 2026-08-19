"""SceneBot policy tracking the box pickup in full physics.

The vendored SceneBot motion graph streams a 50 Hz reference (walk -> squat
pickup -> carry -> reverse put-down); the SceneBot ONNX policy tracks it with
torque PD at 200 Hz on their 29-DoF flat-hand G1. The box is a plain free
body picked up by hand friction alone -- exactly the web demo, minus the
browser.

A scripted commander replaces the demo's keyboard: settle, walk to the box,
pick it up (auto upper-body freeze at completion), carry it forward, put it
down, walk away. The reference stream is open loop, so a physics-free dry
run of the same script predicts where the reference will stand when the
pickup starts, and the box is spawned at exactly the spot the pickup clip
reaches for.

Exit code 0 only if the box was lifted, placed back upright at rest height,
and the robot walked away without falling.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
os.environ.setdefault('MUJOCO_GL',
                      'glfw' if '--viewer' in sys.argv else 'egl')

import json

import numpy as np
import mujoco

from scenebot_tracking import params as P
from scenebot_tracking import motion_graph as MG
from scenebot_tracking.clips import ClipBundle, ContactLabels
from scenebot_tracking.motion_graph import (MotionGraphPlayer, UpperBodyFreeze,
                                            ref_qpos36)
from scenebot_tracking.policy import ScenebotPolicy
from scenebot_tracking.rotations import (quat_conj_xyzw, quat_rotate_xyzw,
                                         quat_rotate_xyzw as rot_xyzw,
                                         yaw_from_wxyz)

import contact_viz

POLICY_FPS = 1.0 / P.CONTROL_DT
BOX_HALF_Z = 0.15
BOX_REST_Z = BOX_HALF_Z
GHOST_RGBA = np.array([1.0, 0.75, 0.2, 0.55], np.float32)
_EYE3 = np.eye(3).ravel()


class Commander:
    """Deterministic script over graph state; same run dry and in physics."""

    def __init__(self, args):
        self.a = args
        self.phase = 'SETTLE'
        self.tick = 0
        self.phase_start = 0
        self.picked = False
        self.placed = False
        self.pick_idle_pose = None       # (root_pos_w, root_quat_wxyz) at G

    def _elapsed(self):
        return (self.tick - self.phase_start) * P.CONTROL_DT

    def _goto(self, phase):
        self.phase = phase
        self.phase_start = self.tick

    def step(self, graph, freeze_active, last_pkt):
        """Returns (command, clear_freeze). Called once per 50 Hz tick."""
        cmd = MG.STOP
        clear_freeze = False
        ph = self.phase
        if ph == 'SETTLE':
            if self._elapsed() >= self.a.settle_seconds:
                self._goto('WALK')
        elif ph == 'WALK':
            if self._elapsed() < self.a.walk_seconds:
                cmd = MG.FORWARD
            elif graph.is_idle():
                self._goto('PICK')
        elif ph == 'PICK':
            cmd = MG.PICK_UP_BOX
            self.pick_idle_pose = (last_pkt['root_pos_w'].copy(),
                                   last_pkt['root_quat_wxyz'].copy())
            self._goto('WAIT_PICKED')
        elif ph == 'WAIT_PICKED':
            if freeze_active:
                self.picked = True
                self._goto('HOLD')
        elif ph == 'HOLD':
            if self._elapsed() >= self.a.hold_seconds:
                self._goto('CARRY_WALK')
        elif ph == 'CARRY_WALK':
            if self._elapsed() < self.a.carry_seconds:
                cmd = MG.FORWARD
            elif graph.is_idle():
                self._goto('PLACE')
        elif ph == 'PLACE':
            cmd = MG.PUT_DOWN_BOX
            clear_freeze = True
            self._goto('WAIT_PLACED')
        elif ph == 'WAIT_PLACED':
            if graph.state.edge_key == f'{MG.STOP}->{MG.STOP}' \
                    and graph.is_idle():
                self.placed = True
                self._goto('RETREAT')
        elif ph == 'RETREAT':
            if self._elapsed() < self.a.retreat_seconds:
                cmd = MG.BACKWARD
            elif graph.is_idle():
                self._goto('DONE')
        self.tick += 1
        return cmd, clear_freeze

    def done(self):
        return self.phase == 'DONE'


def make_graph():
    with open(P.MOTION_GRAPH) as f:
        graph_json = json.load(f)
    bundle = ClipBundle(P.CLIPS_BIN, P.CLIPS_INDEX)
    contacts = ContactLabels(P.CONTACT_BIN, P.CONTACT_INDEX)
    return MotionGraphPlayer(graph_json, bundle, contacts, P.META), bundle


def grab_point_local(bundle):
    """Where the pickup clip's hands close, in the clip's own world frame."""
    seg_end = 120
    clip = bundle.clip(11)
    mids = 0.5 * (clip.body_pos[:seg_end + 1, 28]
                  + clip.body_pos[:seg_end + 1, 29])
    f = int(np.argmin(mids[:, 2]))
    return mids[f], f


def dry_run(args, bundle):
    """Run the command script on the graph alone to find the pick pose."""
    graph, _ = make_graph()
    freeze = UpperBodyFreeze()
    commander = Commander(args)
    pkt = graph.step(MG.STOP)
    max_ticks = int(args.max_seconds * POLICY_FPS)
    for _ in range(max_ticks):
        cmd, clear = commander.step(graph, freeze.active, pkt)
        pkt = graph.step(cmd)
        if pkt['pickup_forward_completed'] and not freeze.active:
            freeze.snapshot(pkt)
        if clear:
            freeze.clear()
        if commander.done():
            break
    if commander.pick_idle_pose is None:
        raise RuntimeError('dry run never reached the pickup')
    return commander.pick_idle_pose


def box_spawn_from_pick_pose(bundle, pick_pos, pick_quat_wxyz,
                             rest_z=BOX_REST_Z):
    """Align pickup-clip frame 0 to the idle pose; the box goes where the
    clip's hands close, dropped to rest height on the floor."""
    clip = bundle.clip(11)
    r_delta, t = MG.plan_alignment(clip.body_pos[0, 0], clip.body_quat[0, 0],
                                   pick_pos, pick_quat_wxyz)
    grab, f = grab_point_local(bundle)
    spot = rot_xyzw(r_delta, grab) + t
    dyaw = 2.0 * np.arctan2(r_delta[2], r_delta[3])
    clip_yaw = yaw_from_wxyz(clip.body_quat[0, 0])
    box_yaw = dyaw + clip_yaw
    box_quat = np.array([np.cos(box_yaw / 2), 0.0, 0.0, np.sin(box_yaw / 2)])
    return np.array([spot[0], spot[1], rest_z]), box_quat, f


def build_model(args):
    """The floor scene with the box geom re-shaped per --box-* args."""
    spec = mujoco.MjSpec.from_file(P.SCENE_FLOOR_XML)
    geom = next(g for g in spec.geoms if g.name == 'free_box_geom')
    hx, hy, hz = [float(v) for v in args.box_size]
    if args.box_type == 'box':
        geom.size = [hx, hy, hz]
        rest_z, max_half = hz, max(hx, hy, hz)
    elif args.box_type == 'cylinder':
        geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        geom.size = [hx, hz, 0.0]            # radius, half height
        rest_z, max_half = hz, max(hx, hz)
    else:                                    # sphere
        geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        geom.size = [hx, 0.0, 0.0]
        rest_z, max_half = hx, hx
    geom.mass = float(args.box_mass)
    model = spec.compile()
    if getattr(args, 'save_mjcf', None):
        with open(args.save_mjcf, 'w') as f:
            f.write(spec.to_xml())
    return model, rest_z, max_half


class Demo:
    def __init__(self, args):
        self.args = args
        self.model, self.box_rest_z, self.box_max_half = build_model(args)
        self.model.opt.timestep = P.SIM_DT
        self.data = mujoco.MjData(self.model)
        m = self.model

        free_types = {mujoco.mjtJoint.mjJNT_FREE}
        hinge = [j for j in range(m.njnt)
                 if m.jnt_type[j] not in free_types
                 and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                 != 'free_box_joint']
        assert len(hinge) == 29, f'expected 29 robot joints, got {len(hinge)}'
        self.q_at = np.array([m.jnt_qposadr[j] for j in hinge])
        self.dq_at = np.array([m.jnt_dofadr[j] for j in hinge])
        fb = m.joint('floating_base_joint')
        self.rq = int(fb.qposadr[0])
        self.rd = int(fb.dofadr[0])
        bx = m.joint('free_box_joint')
        self.bq = int(bx.qposadr[0])
        self.bd = int(bx.dofadr[0])
        act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                     for i in range(m.nu)]
        assert m.nu == 29, act_names

        self.graph, self.bundle = make_graph()
        self.freeze = UpperBodyFreeze()
        self.commander = Commander(args)
        self.policy = ScenebotPolicy()

        pick_pos, pick_quat = dry_run(args, self.bundle)
        self.box_spawn, self.box_spawn_quat, grab_f = \
            box_spawn_from_pick_pose(self.bundle, pick_pos, pick_quat,
                                     self.box_rest_z)
        print(f'[setup] reference pick pose xy=({pick_pos[0]:.2f}, '
              f'{pick_pos[1]:.2f}) yaw={np.degrees(yaw_from_wxyz(pick_quat)):.0f} deg; '
              f'box spawned at ({self.box_spawn[0]:.2f}, '
              f'{self.box_spawn[1]:.2f}) (clip grab frame {grab_f})')

        d = self.data
        d.qpos[:] = 0.0
        d.qpos[self.rq:self.rq + 7] = P.INIT_QPOS_36[0:7]
        d.qpos[self.q_at] = P.INIT_QPOS_36[7:36]
        d.qpos[self.bq:self.bq + 3] = self.box_spawn
        d.qpos[self.bq + 3:self.bq + 7] = self.box_spawn_quat
        mujoco.mj_forward(self.model, d)

        self.pkt = self.graph.step(MG.STOP)
        self.t = 0.0
        self.max_box_z = 0.0
        self.lift_seen = False
        self.dropped = False
        self.max_xy_err = 0.0
        self._last_frame = None
        self.fallen, self.fall_time = False, None
        self.ref_qpos = ref_qpos36(self.pkt['joint_pos_isaac'],
                                   self.pkt['root_pos_w'],
                                   self.pkt['root_quat_wxyz'],
                                   P.ISAAC_TO_MUJOCO)

        self.gdata = mujoco.MjData(self.model)
        self.ghost_bodies = [
            b for b in range(1, self.model.nbody)
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            != 'free_box']

        self.contact_sites = contact_viz.resolve_sites(self.model)
        self._open_outputs()
        self.viewer = None
        if args.viewer:
            import mujoco.viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance, self.viewer.cam.azimuth = 3.6, -35.0
            self.viewer.cam.elevation = -16.0
            self.viewer.cam.lookat[:] = [0.8, 0.0, 0.7]

    def _read_state(self):
        d = self.data
        quat_wxyz = d.qpos[self.rq + 3:self.rq + 7]
        return {
            'q': d.qpos[self.q_at].copy(),
            'dq': d.qvel[self.dq_at].copy(),
            'root_pos': d.qpos[self.rq:self.rq + 3].copy(),
            'root_orn_xyzw': np.array([quat_wxyz[1], quat_wxyz[2],
                                       quat_wxyz[3], quat_wxyz[0]]),
            'root_vel': d.qvel[self.rd:self.rd + 3].copy(),
            'omega': d.qvel[self.rd + 3:self.rd + 6].copy(),
        }

    def step_control(self):
        cmd, clear = self.commander.step(self.graph, self.freeze.active,
                                         self.pkt)
        pkt = self.graph.step(cmd)
        if pkt['pickup_forward_completed'] and not self.freeze.active:
            self.freeze.snapshot(pkt)
            print(f'[{self.t:6.2f}s] PICKUP COMPLETE -> upper body frozen')
        if clear:
            self.freeze.clear()
            print(f'[{self.t:6.2f}s] PUT DOWN -> upper body released')
        self.freeze.apply_to_packet(pkt)
        self.pkt = pkt
        self.ref_qpos = ref_qpos36(
            self.freeze.blend_ghost_joints(pkt['joint_pos_isaac']),
            pkt['root_pos_w'], pkt['root_quat_wxyz'], P.ISAAC_TO_MUJOCO)
        self.policy.ingest(pkt)

        target = self.policy.step(self._read_state())
        d = self.data
        for _ in range(P.DECIMATION):
            tau = (P.KPS * (target - d.qpos[self.q_at])
                   - P.KDS * d.qvel[self.dq_at])
            d.ctrl[:29] = np.clip(tau, -P.TORQUE_LIMIT, P.TORQUE_LIMIT)
            mujoco.mj_step(self.model, d)

        box_z = float(d.qpos[self.bq + 2])
        self.max_box_z = max(self.max_box_z, box_z)
        if box_z > self.box_rest_z + 0.25:
            self.lift_seen = True
        if (self.lift_seen and self.freeze.active
                and box_z < self.box_rest_z + 0.05):
            if not self.dropped:
                print(f'[{self.t:6.2f}s] BOX DROPPED mid-carry')
            self.dropped = True
        xy_err = float(np.linalg.norm(
            d.qpos[self.rq:self.rq + 2] - self.pkt['root_pos_w'][0:2]))
        self.max_xy_err = max(self.max_xy_err, xy_err)

    def _check_fall(self):
        # The squat pickup pitches the torso close to horizontal on purpose,
        # so only a genuine topple counts: pelvis near the floor, or tilted
        # past horizontal.
        down = np.array([0.0, 0.0, -1.0])
        quat_wxyz = self.data.qpos[self.rq + 3:self.rq + 7]
        q_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3],
                           quat_wxyz[0]])
        gravity = quat_rotate_xyzw(quat_conj_xyzw(q_xyzw), down)
        root_z = float(self.data.qpos[self.rq + 2])
        if not self.fallen and (root_z < 0.30 or gravity[2] > 0.1):
            self.fallen, self.fall_time = True, self.t
            print(f'[{self.t:6.2f}s] ROBOT FELL')

    def _open_outputs(self):
        a = self.args
        self.renderer = self.writer = self.cam = None
        self.look = np.array([0.6, 0.0, 0.6])
        if a.no_video:
            return
        import imageio
        os.makedirs(os.path.dirname(a.video_path), exist_ok=True)
        self.renderer = mujoco.Renderer(self.model, a.height, a.width)
        self.writer = imageio.get_writer(a.video_path, fps=int(POLICY_FPS),
                                         codec='libx264', quality=8,
                                         macro_block_size=None)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.azimuth, self.cam.elevation = \
            3.4, -35.0, -16.0

    def _draw_ghost(self, scn):
        g = self.gdata
        g.qpos[:] = 0.0
        g.qpos[self.rq + 3] = 1.0
        g.qpos[self.rq:self.rq + 7] = self.ref_qpos[0:7]
        g.qpos[self.q_at] = self.ref_qpos[7:36]
        mujoco.mj_kinematics(self.model, g)
        for b in self.ghost_bodies:
            pa = self.model.body_parentid[b]
            if pa == 0 or scn.ngeom >= scn.maxgeom:
                continue
            a, c = g.xpos[pa], g.xpos[b]
            if np.linalg.norm(c - a) < 1e-6:
                continue
            gm = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(gm, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                np.zeros(3), np.zeros(3), _EYE3, GHOST_RGBA)
            mujoco.mjv_connector(gm, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.025,
                                 np.asarray(a, float), np.asarray(c, float))
            scn.ngeom += 1

    def render(self):
        if self.renderer is None:
            return
        box_xy = self.data.qpos[self.bq:self.bq + 2]
        focus = np.array([0.6 * self.data.qpos[self.rq] + 0.4 * box_xy[0],
                          0.6 * self.data.qpos[self.rq + 1] + 0.4 * box_xy[1],
                          0.6])
        self.look += 0.06 * (focus - self.look)
        self.cam.lookat[:] = self.look
        self.renderer.update_scene(self.data, camera=self.cam)
        self._draw_ghost(self.renderer.scene)
        contact_viz.draw(self.renderer.scene, self.data,
                         self.contact_sites, self.policy.contact_mask)
        frame = self._annotate(self.renderer.render())
        self._last_frame = frame
        self.writer.append_data(frame)

    def _caption(self):
        a = self.args
        if a.box_type == 'box':
            dims = ' x '.join(f'{2 * v:.2f}' for v in a.box_size)
        elif a.box_type == 'cylinder':
            dims = f'r {a.box_size[0]:.2f}, h {2 * a.box_size[2]:.2f}'
        else:
            dims = f'r {a.box_size[0]:.2f}'
        return f'{a.box_type} {dims} m · {a.box_mass:g} kg'

    def _annotate(self, frame):
        from PIL import Image, ImageDraw, ImageFont
        im = Image.fromarray(frame)
        draw = ImageDraw.Draw(im, 'RGBA')
        font = ImageFont.load_default(size=max(13, im.height // 26))
        text = self._caption()
        box = draw.textbbox((0, 0), text, font=font)
        pad = 6
        draw.rectangle([8, im.height - box[3] - 2 * pad - 8,
                        8 + box[2] + 2 * pad, im.height - 8],
                       fill=(252, 252, 251, 200))
        draw.text((8 + pad, im.height - box[3] - pad - 8), text,
                  fill=(11, 11, 11, 255), font=font)
        return np.asarray(im)

    def run(self):
        a = self.args
        wall = time.time()
        for tick in range(int(a.max_seconds * POLICY_FPS)):
            self.t = tick * P.CONTROL_DT
            self.step_control()
            self._check_fall()
            self.render()
            if self.viewer is not None:
                if not self.viewer.is_running():
                    break
                scn = getattr(self.viewer, 'user_scn', None)
                if scn is not None:
                    scn.ngeom = 0
                    self._draw_ghost(scn)
                    contact_viz.draw(scn, self.data, self.contact_sites,
                                     self.policy.contact_mask)
                self.viewer.sync()
            if self.fallen and (self.t - self.fall_time) > 2.0:
                break
            if self.commander.done():
                break
            if tick % 100 == 0:
                print(f'  t={self.t:5.1f}s phase={self.commander.phase:12s} '
                      f'cmd={self.graph.state.command:16s} '
                      f'x={self.data.qpos[self.rq]:5.2f} '
                      f'box_z={self.data.qpos[self.bq + 2]:5.2f}', flush=True)
        return self.finish(wall)

    def finish(self, wall):
        d = self.data
        box_z = float(d.qpos[self.bq + 2])
        # placed = back at floor level (any resting face) and not riding
        # the robot; the exact face is shape-dependent, so judge by height
        placed_now = box_z < self.box_max_half + 0.08
        away = float(np.linalg.norm(d.qpos[self.rq:self.rq + 2]
                                    - d.qpos[self.bq:self.bq + 2]))
        ok = (self.lift_seen and self.commander.picked
              and self.commander.placed and placed_now
              and not self.dropped and not self.fallen and away > 0.8)
        if self.writer is not None:
            self.writer.close()
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
        print(f'[scenebot-pickup] {"SUCCESS" if ok else "INCOMPLETE"} -- '
              f'{self.t:.1f} s sim, {time.time() - wall:.0f} s wall, '
              f'lifted={self.lift_seen} (max z {self.max_box_z:.2f} m), '
              f'dropped={self.dropped}, placed={placed_now} (z {box_z:.2f}), '
              f'fallen={self.fallen}, robot-box dist {away:.2f} m, '
              f'phase={self.commander.phase}')
        print('[stress-json] ' + json.dumps({
            'box_type': self.args.box_type,
            'box_size': [float(v) for v in self.args.box_size],
            'box_mass': float(self.args.box_mass),
            'success': bool(ok), 'lifted': bool(self.lift_seen),
            'dropped': bool(self.dropped), 'placed': bool(placed_now),
            'fallen': bool(self.fallen),
            'picked_cmd': bool(self.commander.picked),
            'placed_cmd': bool(self.commander.placed),
            'max_box_z': round(self.max_box_z, 3),
            'final_box_z': round(box_z, 3),
            'max_xy_err': round(self.max_xy_err, 3),
            'away': round(away, 2), 'sim_s': round(self.t, 1)}))
        if self.writer is not None:
            print(f'[scenebot-pickup] wrote {self.args.video_path}')
        return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--viewer', action='store_true')
    ap.add_argument('--max-seconds', type=float, default=45.0)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--settle-seconds', type=float, default=1.5)
    ap.add_argument('--walk-seconds', type=float, default=3.0)
    ap.add_argument('--hold-seconds', type=float, default=1.0)
    ap.add_argument('--carry-seconds', type=float, default=2.5)
    ap.add_argument('--retreat-seconds', type=float, default=2.5)
    ap.add_argument('--box-type', choices=['box', 'cylinder', 'sphere'],
                    default='box')
    ap.add_argument('--box-size', type=float, nargs=3,
                    default=[0.15, 0.10, 0.15],
                    help='half extents (m); cylinder uses [radius, -, half '
                         'height], sphere uses [radius, -, -]')
    ap.add_argument('--box-mass', type=float, default=0.1)
    ap.add_argument('--save-mjcf', default=None,
                    help='write the compiled scene MJCF (with the box '
                         'dimensions) to this path')
    ap.add_argument('--video',
                    default=os.path.join(HERE, 'out', 'scenebot_pickup.mp4'))
    args = ap.parse_args()
    args.video_path = args.video
    return Demo(args).run()


if __name__ == '__main__':
    sys.exit(main())
