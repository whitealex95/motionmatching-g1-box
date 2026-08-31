"""SONIC 29-DoF G1 scene + the box, assembled in code (modes: kinematic /
grasp).

The carton mesh sits tilted inside its own local frame on purpose -- the
OmniRetarget clips' box quaternion is calibrated to the scanned box's frame
(see tools/make_medicine_box.py) -- so the true oriented box (centre, axes,
half extents) is recovered from the mesh corners for the ghost overlay.
"""
import os
import sys
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mm_g1 import config as C
BOX_MESH = os.path.join(ROOT, 'assets', 'largebox', 'medicinebox.obj')
BOX_TEX = os.path.join(ROOT, 'assets', 'largebox', 'medicinebox.png')

PALM_FRICTION = [1.5, 0.02, 0.0005]


def _carton_obb():
    """(centre, axes 3x3 columns, half extents) of the carton in its local frame."""
    v = np.array([[float(x) for x in l.split()[1:4]]
                  for l in open(BOX_MESH) if l.startswith('v ')])
    corners = np.unique(np.round(v, 6), axis=0)          # (8, 3)
    center = corners.mean(0)
    cov = (corners - center).T @ (corners - center) / len(corners)
    w, R = np.linalg.eigh(cov)                           # eigvals = half^2
    return center, R, np.sqrt(w)


GHOST_CENTER, GHOST_MAT, GHOST_HALF = _carton_obb()


def build_model(scene_xml_path, mode, box_mass=0.5, off_w=1280, off_h=720,
                box_scale=1.0, box_type='scenebot'):
    """box_type 'scenebot' (default) = the SceneBot free box with the
    C.BOX_HALF extents the motion library is baked for; 'carton' = the
    OmniRetarget medicine-box mesh (a DIFFERENT size than the library's box)."""
    spec = mujoco.MjSpec.from_file(scene_xml_path)

    # SONIC's rubber hands are visual-only: bolt a contact pad onto each
    # wrist. Robots that already have palm sites (SceneBot's flat-hand G1)
    # keep their own hand collision.
    if not any(s.name == 'left_palm' for s in spec.sites):
        for side, ysgn in (('left', 1.0), ('right', -1.0)):
            wrist = spec.body(f'{side}_wrist_yaw_link')
            wrist.add_geom(name=f'{side}_palm_col',
                           type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                           fromto=[0.04, ysgn * 0.007, 0.0,
                                   0.135, ysgn * 0.007, 0.0],
                           size=[0.022, 0.0, 0.0], rgba=[0.7, 0.7, 0.7, 1.0],
                           friction=PALM_FRICTION)
            wrist.add_site(name=f'{side}_palm', pos=[0.10, ysgn * 0.007, 0.0],
                           size=[0.012] * 3, rgba=[0.1, 0.9, 0.1, 0.35])

    collide = mode != 'kinematic'
    if box_type == 'carton':
        tex = spec.add_texture()
        tex.name = 'box_tex'
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.file = BOX_TEX
        mat = spec.add_material()
        mat.name = 'box_mat'
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'box_tex'
        mat.texuniform = False
        # box_scale shrinks/grows only the physical carton (about the mesh
        # origin ~= its centre; UVs ride along, so the texture is unchanged).
        # The clips' box pose and the reference-box ghost stay data-sized.
        mesh = spec.add_mesh(name='box_mesh', file=BOX_MESH)
        mesh.scale = [float(box_scale)] * 3
        box = spec.worldbody.add_body(name='largebox', pos=[1.6, 0.0, 0.19])
        box.add_joint(name='box_joint', type=mujoco.mjtJoint.mjJNT_FREE)
        box.add_geom(name='box_geom', type=mujoco.mjtGeom.mjGEOM_MESH,
                     meshname='box_mesh', material='box_mat',
                     mass=float(box_mass), friction=PALM_FRICTION,
                     contype=1 if collide else 0,
                     conaffinity=1 if collide else 0)
        ghost = (GHOST_CENTER, GHOST_MAT, GHOST_HALF)
    else:                                            # scenebot free box
        half = np.array(C.BOX_HALF, float) * float(box_scale)
        box = spec.worldbody.add_body(name='largebox',
                                      pos=[1.6, 0.0, float(half[2])])
        box.add_joint(name='box_joint', type=mujoco.mjtJoint.mjJNT_FREE)
        box.add_geom(name='box_geom', type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=half, rgba=[0.82, 0.52, 0.22, 1.0],
                     mass=float(box_mass), friction=PALM_FRICTION,
                     contype=1 if collide else 0,
                     conaffinity=1 if collide else 0)
        ghost = (np.zeros(3), np.eye(3), np.array(C.BOX_HALF, float))

    model = spec.compile()
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, off_w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, off_h)

    ids = dict(
        box_qpos_at=model.joint('box_joint').qposadr[0],
        box_dof_at=model.joint('box_joint').dofadr[0],
        box_body=model.body('largebox').id,
        ghost=ghost,                      # (centre, axes, half extents) in box frame
    )
    return model, ids
