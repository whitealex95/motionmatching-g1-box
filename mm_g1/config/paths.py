"""Repo-relative locations of data, scenes, and the library cache.

Everything is resolved relative to this file so the project is fully relocatable --
clone the folder anywhere and the model, data and cache paths still line up.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data", "gmr_lafan1_g1")   # GMR-retargeted LAFAN1 .pkl clips
BOX_DATA_DIR = os.path.join(ROOT, "data", "robot_object_g1")  # OmniRetarget pick/carry/place .npz
SCENE_XML = os.path.join(ROOT, "assets", "unitree_g1", "scene.xml")       # G1 only (FK + loco)
SCENE_BOX_XML = os.path.join(ROOT, "assets", "unitree_g1", "scene_box.xml")  # G1 + box (interactive)
SCENE_BOX_SCENEBOT_XML = os.path.join(ROOT, "assets", "unitree_g1", "scene_box_scenebot.xml")
LIB_PATH = os.path.join(ROOT, "data", "motion_lib.npz")   # built on first run, then cached
