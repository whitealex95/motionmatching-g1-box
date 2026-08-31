"""Headless regression test of the single-pick kinematic demo.

Drives the matcher exactly like the viewer does (30 Hz, no physics): spawn the
box off to the side at an angle, press B, and assert the whole chain runs:
MOVE-TO-PICK walks the planned route to the stance, PICK rides the baked
SceneBot squat, CARRY holds the box (OmniRetarget frames), B again rides the
reversed drop, and the box ends back at rest with the robot in locomotion.

    python -m mm_g1.test_headless [--video out.mp4]
"""
import argparse
import sys

import numpy as np

from . import config as C
from .controller import MotionMatcher
from .data import load_library
from .features import yaw_quat


def run(video=None):
    lib = load_library()
    m = MotionMatcher(lib)

    # Box off the start axis and turned: the approach must corner and pick a fold.
    m.boxPos = np.array([1.7, 0.9, C.BOX_REST_Z])
    m.boxRot = yaw_quat(np.deg2rad(40.0))
    m.boxPosPrev = m.boxPos.copy()

    frames = []
    render = None
    if video:
        import mujoco

        model = mujoco.MjModel.from_xml_path(C.SCENE_BOX_SCENEBOT_XML)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, 720, 960)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 160.0, -18.0, 3.4

        def render(q):
            data.qpos[0:36] = q
            data.qpos[36:43] = m.box_qpos()
            mujoco.mj_forward(model, data)
            mid = 0.5 * (q[0:2] + m.boxPos[0:2])
            cam.lookat[:] = [mid[0], mid[1], 0.7]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    seen = set()
    max_box_z = 0.0
    stance_err_at_pick = None
    prev_state = m.state_name()

    def step_until(want, timeout_s, vel=None, face=None):
        nonlocal max_box_z, stance_err_at_pick, prev_state
        for _ in range(int(timeout_s / C.DT)):
            q = m.step(np.zeros(3) if vel is None else vel,
                       np.zeros(3) if face is None else face)
            assert np.isfinite(q).all(), "NaN in qpos"
            if render:
                render(q)
            st = m.state_name()
            seen.add(st)
            if st == "PICK" and prev_state == "MOVE-TO-PICK":
                stance_err_at_pick = float(
                    np.linalg.norm(m.rootPos[0:2] - m.stance_xy))
            prev_state = st
            max_box_z = max(max_box_z, float(m.boxPos[2]))
            if st == want:
                return True
        return False

    assert step_until("LOCOMOTION", 1.0), "did not settle"
    m.trigger_box()
    assert step_until("MOVE-TO-PICK", 1.0), "B did not start the approach"
    assert step_until("PICK", C.MOVE_TIMEOUT + 2.0), "approach never reached the stance"
    assert step_until("CARRY", 8.0), "pick ride did not hand off to carry"
    assert m.box_held, "box not held after pick"
    fwd = np.array([1.0, 0.0, 0.0])
    step_until("__never__", 2.0, vel=0.5 * fwd)          # carry-walk 2 s
    m.trigger_box()
    assert step_until("PLACE", 2.0), "B while carrying did not enter place"
    assert step_until("LOCOMOTION", 6.0), "place ride did not finish"
    assert not m.box_held, "box still held after place"
    step_until("__never__", 1.5)                          # settle out

    box_z = float(m.boxPos[2])
    assert max_box_z > 0.5, f"box never lifted (max z {max_box_z:.2f})"
    assert abs(box_z - C.BOX_REST_Z) < 0.03, f"box not back at rest (z {box_z:.2f})"
    assert stance_err_at_pick is not None and stance_err_at_pick < 0.25, \
        f"pick entered {stance_err_at_pick} m off the stance"

    if video:
        import imageio

        w = imageio.get_writer(video, fps=C.FPS, codec="libx264", quality=8,
                               macro_block_size=1)
        for f in frames:
            w.append_data(f)
        w.close()
        print(f"wrote {video} ({len(frames)} frames)")

    print(f"OK: states {sorted(seen)}, max box z {max_box_z:.2f} m, "
          f"final box z {box_z:.2f} m, stance err at pick "
          f"{stance_err_at_pick:.3f} m")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    sys.exit(run(ap.parse_args().video))
