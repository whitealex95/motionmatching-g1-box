"""Offscreen kinematic playback of a G1 qpos clip (+ optional box) to mp4.

    MUJOCO_GL=egl python tools/omniretarget_emm/render_clip.py clip.npz out.mp4 [--box-key box_pose]

Reads `qpos` (T, 36) and, if present, `box_pose` (T, 7) from the npz. Works for the
GMR clips in data/emm_g1/clips too (no box there: pass --wrist-box to draw the
0.3 x 0.2 x 0.3 box at the wrist midpoint, the way mm_g1/emm_clips.py synthesizes it).
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SCENE = os.path.join(ROOT, "assets", "unitree_g1", "scene_box_scenebot.xml")


def _yaw_quat(yaw):
    return np.stack([np.cos(0.5 * yaw), 0 * yaw, 0 * yaw, np.sin(0.5 * yaw)], -1)


def wrist_box(model, data, qpos, rest_z=0.15):
    """(T, 7) box at the wrist midpoint, yawed with the root (mm_g1/emm_clips.py)."""
    import mujoco
    ids = [model.body("left_wrist_yaw_link").id, model.body("right_wrist_yaw_link").id]
    out = np.zeros((len(qpos), 7))
    for t, q in enumerate(qpos):
        data.qpos[:36] = q
        mujoco.mj_kinematics(model, data)
        mid = data.xpos[ids].mean(0)
        out[t, :3] = mid
        out[t, 2] = max(mid[2], rest_z)
        w, x, y, z = q[3:7]
        out[t, 3:] = _yaw_quat(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return out


def render(qpos, box_pose, out_path, fps=30, box_size=(0.3, 0.2, 0.3), width=960, height=720,
           side_by_side_with=None, quality=8):
    import imageio
    import mujoco

    model = mujoco.MjModel.from_xml_path(SCENE)
    gid = model.geom("box_geom").id
    model.geom_size[gid] = 0.5 * np.asarray(box_size)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 3.2, -18.0, 135.0

    clips = [(qpos, box_pose)] + ([side_by_side_with] if side_by_side_with else [])
    n = min(len(c[0]) for c in clips)
    with imageio.get_writer(out_path, fps=int(fps), codec="libx264", quality=quality,
                            macro_block_size=1) as w:
        for t in range(n):
            frames = []
            for q, bp in clips:
                data.qpos[:36] = q[t]
                if bp is not None:
                    data.qpos[36:43] = bp[t]
                else:
                    data.qpos[36:43] = [0, 0, -1, 1, 0, 0, 0]          # hide below the floor
                mujoco.mj_forward(model, data)
                cam.lookat[:] = data.qpos[:3] + [0, 0, -0.1]
                renderer.update_scene(data, cam)
                frames.append(renderer.render())
            w.append_data(np.concatenate(frames, 1) if len(frames) > 1 else frames[0])
    renderer.close()
    print(f"wrote {out_path} ({n} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--wrist-box", action="store_true", help="draw a box at the wrist midpoint")
    ap.add_argument("--compare", default=None, help="second npz rendered side by side (right)")
    ap.add_argument("--box-size", type=float, nargs=3, default=(0.3, 0.2, 0.3))
    ap.add_argument("--size", type=int, nargs=2, default=(960, 720), help="frame width height")
    ap.add_argument("--quality", type=int, default=8, help="imageio/ffmpeg quality 0..10")
    args = ap.parse_args()

    import mujoco
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)

    def load(path):
        d = np.load(path)
        q = d["qpos"].astype(np.float64)
        if "box_pose" in d:
            bp = d["box_pose"].astype(np.float64)
        elif args.wrist_box:
            bp = wrist_box(model, data, q)
        else:
            bp = None
        return q, bp, int(d["fps"]) if "fps" in d else 30

    q, bp, fps = load(args.npz)
    other = load(args.compare)[:2] if args.compare else None
    render(q, bp, args.out, fps=fps, box_size=args.box_size, side_by_side_with=other,
           width=args.size[0], height=args.size[1], quality=args.quality)


if __name__ == "__main__":
    main()
