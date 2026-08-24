#!/usr/bin/env python3
"""Interactive, keyboard-controlled motion matching for the Unitree G1.

    python run.py                # the plain box
    python run.py --medicine     # the printed 'MEDICINE' carton (same box, different skin)

Builds (or loads the cached) motion library, opens a MuJoCo window, and lets you steer the
G1 around with WASD in real time and pick up / carry / set down a box with B. The first
launch spends ~20-40 s building the feature database (data/motion_lib.npz); later launches
start instantly.

Controls
  W / A / S / D ........ move, relative to the camera
  Arrow keys ........... face direction, independent of travel (GenoView-style)
  Shift (hold) ......... walk instead of run (full stick is run pace, GenoView-style)
  B .................... box action: pick up when near the box, set down while carrying
  Space ................ reset to the start pose
  Left-drag / right-drag / scroll ... orbit / pan / zoom
  Esc .................. quit
"""
import argparse
import mujoco

from mm_g1 import config as C
from mm_g1.data import load_library
from mm_g1.controller import MotionMatcher
from mm_g1.viewer import InteractiveViewer


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-only", action="store_true",
                    help="build/refresh the motion library cache and exit (no window)")
    ap.add_argument("--medicine", action="store_true",
                    help="carry the printed 'MEDICINE' carton instead of the plain box "
                         "(visual swap only -- same box volume, same motion)")
    args = ap.parse_args()

    print("Loading motion library (first run builds the feature cache)...")
    lib = load_library()
    print(f"  {len(lib['qpos'])} frames, clips: {', '.join(map(str, lib['clip_names']))}")

    matcher = MotionMatcher(lib)
    n_search = sum(re - rs for rs, re, _ in matcher.loco_trees + matcher.carry_trees)
    print(f"  feature DB ready ({n_search} searchable loco+carry frames, "
          f"{len(matcher.pick_enter)} pick / {len(matcher.place_enter)} place entries)")
    if args.build_only:
        print("Build complete. Run `python run.py` to control the G1.")
        return

    scene = (C.SCENE_BOX_SCENEBOT_XML if C.SCENEBOT_PICK else
             C.SCENE_BOX_MEDICINE_XML if args.medicine else C.SCENE_BOX_XML)
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    print("Opening viewer -- WASD to move, B to pick up / set down the box, Esc to quit.")
    InteractiveViewer(model, data, matcher).run()


if __name__ == "__main__":
    main()
