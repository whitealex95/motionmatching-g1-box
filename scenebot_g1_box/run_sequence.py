"""The SceneBot demo page's example sequence (the Enter key), in physics.

Replays the demo's token string Q-L-L-E-W-N-G-W-Z-P on their original
pedestal scene: spin left, sit on the bench, stand up, spin right, walk,
step onto the 0.36 m pedestal, grab the box off it, walk across while
carrying (upper body frozen), step down, put the box on the floor.

Token semantics are the demo's example runner: wait until the graph is
idle (segment finished, no pending yaw) and stable for 0.4 s, then tap the
next token. Q/E queue +/-180 deg of reference yaw consumed at 60 deg/s;
L toggles SitDown/StandUp; the rest are one-tick graph commands. The
reference stream is open loop, so this reproduces the web demo's reference
trajectory exactly; only the physical tracking differs.

Exit code 0 only if the robot sat, climbed the pedestal, lifted the box
(> 0.75 m), brought it down, left it resting on the floor, and never fell.
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

import numpy as np
import mujoco

from scenebot_tracking import params as P
from scenebot_tracking import motion_graph as MG
from scenebot_tracking.motion_graph import UpperBodyFreeze, ref_qpos36
from scenebot_tracking.policy import ScenebotPolicy
from scenebot_tracking.rotations import quat_conj_xyzw, quat_rotate_xyzw

from run_pickup import make_graph
import contact_viz

POLICY_FPS = 1.0 / P.CONTROL_DT
YAW_RATE = np.radians(60.0)
SEQUENCE = 'QLLEWNGWZP'
GHOST_RGBA = np.array([1.0, 0.75, 0.2, 0.55], np.float32)
_EYE3 = np.eye(3).ravel()
TOKEN_CMDS = {'W': MG.FORWARD, 'N': MG.STEP_ON_BOX, 'G': MG.PICK_UP_BOX,
              'Z': MG.COME_DOWN_BOX, 'P': MG.PUT_DOWN_BOX}


class TokenSequencer:
    def __init__(self, tokens, start_time=1.5, settle_ticks=20):
        self.tokens = list(tokens)
        self.i = 0
        self.start_time = start_time
        self.settle_ticks = settle_ticks
        self.idle_ticks = 0
        self.yaw_remaining = 0.0
        self.sit_forward_next = True

    def all_dispatched(self):
        return self.i >= len(self.tokens)

    def step(self, graph, t):
        """Returns (command, clear_freeze, yaw_step, dispatched_token)."""
        yaw = float(np.clip(self.yaw_remaining,
                            -YAW_RATE * P.CONTROL_DT,
                            YAW_RATE * P.CONTROL_DT))
        self.yaw_remaining -= yaw
        cmd, clear, tok = MG.STOP, False, None
        pending = abs(self.yaw_remaining) > 1e-9 or yaw != 0.0
        if (not self.all_dispatched() and t >= self.start_time
                and not pending and graph.is_idle()):
            self.idle_ticks += 1
            if self.idle_ticks >= self.settle_ticks:
                self.idle_ticks = 0
                tok = self.tokens[self.i]
                self.i += 1
                cmd, clear = self._dispatch(tok)
        else:
            self.idle_ticks = 0
        return cmd, clear, yaw, tok

    def _dispatch(self, tok):
        if tok == 'Q':
            self.yaw_remaining += np.pi
            return MG.STOP, False
        if tok == 'E':
            self.yaw_remaining -= np.pi
            return MG.STOP, False
        if tok == 'L':
            cmd = MG.SIT_DOWN if self.sit_forward_next else MG.STAND_UP
            self.sit_forward_next = not self.sit_forward_next
            return cmd, False
        return TOKEN_CMDS[tok], tok == 'P'


class Demo:
    def __init__(self, args):
        self.args = args
        self.model = mujoco.MjModel.from_xml_path(P.SCENE_XML)
        self.model.opt.timestep = P.SIM_DT
        self.model.vis.global_.offwidth = max(
            self.model.vis.global_.offwidth, args.width)
        self.model.vis.global_.offheight = max(
            self.model.vis.global_.offheight, args.height)
        self.data = mujoco.MjData(self.model)
        m = self.model

        hinge = [j for j in range(m.njnt)
                 if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        assert len(hinge) == 29, len(hinge)
        self.q_at = np.array([m.jnt_qposadr[j] for j in hinge])
        self.dq_at = np.array([m.jnt_dofadr[j] for j in hinge])
        fb = m.joint('floating_base_joint')
        self.rq = int(fb.qposadr[0])
        self.rd = int(fb.dofadr[0])
        bx = m.joint('free_box_joint')
        self.bq = int(bx.qposadr[0])

        self.graph, _ = make_graph()
        self.freeze = UpperBodyFreeze()
        self.seq = TokenSequencer(SEQUENCE, start_time=args.start_seconds)
        self.policy = ScenebotPolicy()

        d = self.data
        mujoco.mj_resetData(m, d)          # box/pedestal at scene defaults
        d.qpos[self.rq:self.rq + 7] = P.INIT_QPOS_36[0:7]
        d.qpos[self.q_at] = P.INIT_QPOS_36[7:36]
        mujoco.mj_forward(m, d)

        self.pkt = self.graph.step(MG.STOP)
        self.ref_qpos = ref_qpos36(self.pkt['joint_pos_isaac'],
                                   self.pkt['root_pos_w'],
                                   self.pkt['root_quat_wxyz'],
                                   P.ISAAC_TO_MUJOCO)
        self.t = 0.0
        self.sat_root_z = None
        self.max_root_z = 0.0
        self.max_box_z = 0.0
        self.came_down = False
        self.fallen, self.fall_time = False, None
        self.done_time = None

        self.gdata = mujoco.MjData(self.model)
        self.ghost_bodies = [
            b for b in range(1, m.nbody)
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            not in ('free_box', 'step_1', 'step_2')]

        self.contact_sites = contact_viz.resolve_sites(self.model)
        self._open_outputs()
        self.viewer = None
        if args.viewer:
            import mujoco.viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance, self.viewer.cam.azimuth = 4.2, -50.0
            self.viewer.cam.elevation = -16.0
            self.viewer.cam.lookat[:] = [1.0, 0.0, 0.8]

    def _read_state(self):
        d = self.data
        qw = d.qpos[self.rq + 3:self.rq + 7]
        return {
            'q': d.qpos[self.q_at].copy(),
            'dq': d.qvel[self.dq_at].copy(),
            'root_pos': d.qpos[self.rq:self.rq + 3].copy(),
            'root_orn_xyzw': np.array([qw[1], qw[2], qw[3], qw[0]]),
            'root_vel': d.qvel[self.rd:self.rd + 3].copy(),
            'omega': d.qvel[self.rd + 3:self.rd + 6].copy(),
        }

    def step_control(self):
        cmd, clear, yaw, tok = self.seq.step(self.graph, self.t)
        if tok is not None:
            r = self.data.qpos[self.rq:self.rq + 3]
            print(f'[{self.t:6.2f}s] token {tok:>1s}   robot '
                  f'({r[0]:5.2f}, {r[1]:5.2f}, {r[2]:4.2f})  '
                  f'box_z={self.data.qpos[self.bq + 2]:4.2f}', flush=True)
        pkt = self.graph.step(cmd, None, yaw)
        if pkt['pickup_forward_completed'] and not self.freeze.active:
            self.freeze.snapshot(pkt)
            print(f'[{self.t:6.2f}s] PICKUP COMPLETE -> upper body frozen')
        if clear:
            self.freeze.clear()
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

        root_z = float(d.qpos[self.rq + 2])
        box_z = float(d.qpos[self.bq + 2])
        self.max_root_z = max(self.max_root_z, root_z)
        self.max_box_z = max(self.max_box_z, box_z)
        if (self.sat_root_z is None and self.graph.is_idle()
                and self.graph.state.command == MG.SIT_DOWN):
            self.sat_root_z = root_z
            print(f'[{self.t:6.2f}s] SEATED (root z {root_z:.2f} m)')
        if (not self.came_down and self.max_box_z > 0.75
                and box_z > 0.55 and root_z < 0.95):
            self.came_down = True
            print(f'[{self.t:6.2f}s] CAME DOWN carrying '
                  f'(box z {box_z:.2f} m)')

    def _check_fall(self):
        qw = self.data.qpos[self.rq + 3:self.rq + 7]
        q_xyzw = np.array([qw[1], qw[2], qw[3], qw[0]])
        gravity = quat_rotate_xyzw(quat_conj_xyzw(q_xyzw),
                                   np.array([0.0, 0.0, -1.0]))
        root_z = float(self.data.qpos[self.rq + 2])
        if not self.fallen and (root_z < 0.28 or gravity[2] > 0.1):
            self.fallen, self.fall_time = True, self.t
            print(f'[{self.t:6.2f}s] ROBOT FELL')

    def _open_outputs(self):
        a = self.args
        self.renderer = self.writer = self.cam = None
        self.look = np.array([0.0, 0.0, 0.8])
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
            4.2, -50.0, -16.0

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
        focus = np.array([self.data.qpos[self.rq],
                          self.data.qpos[self.rq + 1], 0.8])
        self.look += 0.05 * (focus - self.look)
        self.cam.lookat[:] = self.look
        self.renderer.update_scene(self.data, camera=self.cam)
        self._draw_ghost(self.renderer.scene)
        contact_viz.draw(self.renderer.scene, self.data,
                         self.contact_sites, self.policy.contact_mask)
        self.writer.append_data(self.renderer.render())

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
            if (self.seq.all_dispatched() and self.graph.is_idle()
                    and abs(self.seq.yaw_remaining) < 1e-9):
                if self.done_time is None:
                    self.done_time = self.t
                elif self.t - self.done_time > 1.5:
                    break
        return self.finish(wall)

    def finish(self, wall):
        if self.writer is not None:
            self.writer.close()
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
        d = self.data
        box_z = float(d.qpos[self.bq + 2])
        sat = self.sat_root_z is not None and self.sat_root_z < 0.65
        stepped_up = self.max_root_z > 1.05
        lifted = self.max_box_z > 0.75
        placed = 0.05 < box_z < 0.22          # flat on the floor, any face
        complete = self.seq.all_dispatched() and self.done_time is not None
        ok = (complete and sat and stepped_up and lifted and self.came_down
              and placed and not self.fallen)
        print(f'[scenebot-sequence] {"SUCCESS" if ok else "INCOMPLETE"} -- '
              f'{self.t:.1f} s sim, {time.time() - wall:.0f} s wall, '
              f'complete={complete}, sat={sat} '
              f'(root z {self.sat_root_z if self.sat_root_z is not None else -1:.2f}), '
              f'stepped_up={stepped_up} (max root z {self.max_root_z:.2f}), '
              f'lifted={lifted} (max box z {self.max_box_z:.2f}), '
              f'came_down={self.came_down}, placed={placed} '
              f'(box z {box_z:.2f}), fallen={self.fallen}')
        if self.writer is not None:
            print(f'[scenebot-sequence] wrote {self.args.video_path}')
        return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--viewer', action='store_true')
    ap.add_argument('--max-seconds', type=float, default=90.0)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--start-seconds', type=float, default=1.5)
    ap.add_argument('--video',
                    default=os.path.join(HERE, 'out', 'scenebot_sequence.mp4'))
    args = ap.parse_args()
    args.video_path = args.video
    return Demo(args).run()


if __name__ == '__main__':
    sys.exit(main())
