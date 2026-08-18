"""Contact-prompt overlay: one sphere per prompted link, colored by scene
type.

The SceneBot contact vector is c in {0,1}^10 over (link, scene-type)
pairs, K = {left foot, right foot, left wrist, right wrist, pelvis} x
{terrain, object}. Even slot = terrain, odd slot = object per link, which
is exactly where the demo's 4->8->10 expansion lands its channels: feet on
terrain slots 0/2, wrists on object slots 5/7, the sit channel on pelvis-
terrain slot 8.

GREEN sphere  = this link is prompted to touch TERRAIN
MAGENTA sphere = this link is prompted to touch the OBJECT
No sphere      = no contact prompted (walking feet are zeroed on purpose)
"""
import mujoco
import numpy as np

LINK_SITES = ['left_foot', 'right_foot', 'left_palm', 'right_palm',
              'pelvis']
# big enough to poke out of each link's mesh (the pelvis site is buried at
# the body centre)
LINK_RADII = [0.07, 0.07, 0.055, 0.055, 0.12]
TERRAIN_RGBA = np.array([0.15, 0.90, 0.25, 0.5], np.float32)
OBJECT_RGBA = np.array([0.95, 0.20, 0.95, 0.5], np.float32)
_EYE3 = np.eye(3).ravel()


def resolve_sites(model):
    return [model.site(n).id for n in LINK_SITES]


def draw(scn, data, site_ids, mask):
    if mask is None:
        return
    for i, sid in enumerate(site_ids):
        for j, rgba in ((0, TERRAIN_RGBA), (1, OBJECT_RGBA)):
            if len(mask) <= 2 * i + j or mask[2 * i + j] < 0.5:
                continue
            if scn.ngeom >= scn.maxgeom:
                return
            gm = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(gm, mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([LINK_RADII[i], 0.0, 0.0]),
                                np.asarray(data.site_xpos[sid], float),
                                _EYE3, rgba)
            scn.ngeom += 1
