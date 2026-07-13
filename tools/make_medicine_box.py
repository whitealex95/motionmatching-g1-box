#!/usr/bin/env python3
"""Generate the 'MEDICINE' cardboard carton used by the box demo's medicine variant.

It stands in for largebox.obj in scene_box_medicine.xml (`python run.py --medicine`, and the
/medicine build of the web demo). The default scene keeps the plain scanned box.

Run from the repo root with any env that has numpy + scipy + pillow, e.g.:
    ~/miniconda3/envs/deploy_mujoco/bin/python tools/make_medicine_box.py

Writes assets/largebox/medicinebox.obj (24 verts / 12 tris, UV-mapped) and its texture
assets/largebox/medicinebox.png (a 3x2 atlas of 512px faces: top, bottom, +X, -X, +Y, -Y).

WHY THIS IS NOT JUST A CUBOID AT THE ORIGIN
-------------------------------------------
largebox.obj is a *scanned* box: irregular, open-rimmed, and -- crucially -- it sits ROTATED
inside its own local frame (only 14% of its surface area faces along local x/y/z). The
OmniRetarget clips' box quaternion is calibrated against that tilted frame, which is why the
scan renders upright in the demo. So a replacement mesh must be authored in the same tilted
frame, or it comes out mis-rotated (upside down) and -- if built from the axis-aligned
bounding box of a tilted object -- roughly 40% too big, so it juts out of the robot's hands.

We therefore recover the box's TRUE pose inside the local frame:
  * `source_obb`  -- minimum-volume oriented bounding box over the convex hull. Its six faces
                     account for ~90% of the scan's surface area (the local axes: 14%), and it
                     measures 0.324 x 0.339 x 0.363 m -- the real carton, not its AABB.
  * `up_axis`     -- which OBB axis the clips' resting box quaternion sends to world +Z. That
                     face gets the lid print; without this the carton lands upside down.
The carton is then built in that frame, so it occupies the same volume as the scan and the
pick/carry/place clips grip it identically. The box geometry is purely visual (the matcher
only ever sees the box POSE from the data), so this swap cannot change the motion.

Regenerate rather than hand-edit: the OBJ is machine-written.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from mm_g1 import config as C
from mm_g1 import quat

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # repo root
OUT_DIR = os.path.join(HERE, "assets", "largebox")
SRC_OBJ = os.path.join(OUT_DIR, "largebox.obj")                        # the scan we replace
FONT_DIR = "/usr/share/fonts/truetype/dejavu"

CELL = 512                        # px per face in the atlas
COLS, ROWS = 3, 2
INSET = 3                         # px of UV inset, so bilinear filtering can't bleed cells

KRAFT = (188, 149, 103)           # corrugated-cardboard base
INK = (58, 52, 48)                # printed-stencil dark
RED = (196, 40, 46)               # medical cross

FACE_CELLS = {"top": (0, 0), "bottom": (1, 0), "+x": (2, 0),
              "-x": (0, 1), "+y": (1, 1), "-y": (2, 1)}


# --------------------------------------------------------------------------
# Recover the real carton's frame from the scan
# --------------------------------------------------------------------------
def source_obb(path=SRC_OBJ, n_angles=181):
    """Minimum-volume oriented bounding box of the scan, in its local frame.

    Returns (R, centre, extents): R's COLUMNS are the box axes. Uses the standard result that
    one face of the optimal box lies flush with a face of the convex hull, so we only have to
    sweep the in-plane angle for each hull face.
    """
    V = np.array([[float(x) for x in l.split()[1:4]]
                  for l in open(path) if l.startswith("v ")])
    hull = ConvexHull(V)
    P = V[hull.vertices]

    def frame_from_normal(n):
        n = n / np.linalg.norm(n)
        a = np.array([0.0, 0, 1]) if abs(n @ [0, 0, 1]) < 0.99 else np.array([1.0, 0, 0])
        x = np.cross(a, n); x /= np.linalg.norm(x)
        return np.stack([x, np.cross(n, x), n], axis=1)          # columns

    best = None
    for eq in hull.equations:
        R0 = frame_from_normal(eq[:3])
        Q = P @ R0
        z0, z1 = Q[:, 2].min(), Q[:, 2].max()
        for th in np.linspace(0, np.pi / 2, n_angles):
            c, s = np.cos(th), np.sin(th)
            W = Q[:, :2] @ np.array([[c, -s], [s, c]])
            lo2, hi2 = W.min(0), W.max(0)
            vol = (hi2[0] - lo2[0]) * (hi2[1] - lo2[1]) * (z1 - z0)
            if best is None or vol < best[0]:
                R = R0 @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
                lo = np.array([lo2[0], lo2[1], z0])
                hi = np.array([hi2[0], hi2[1], z1])
                best = (vol, R, lo, hi)
    _, R, lo, hi = best
    return R, R @ ((lo + hi) / 2), hi - lo


def rest_quat():
    """The box's resting orientation (wxyz) at the start of a pick/carry/place clip."""
    name = sorted(f for f in os.listdir(C.BOX_DATA_DIR) if f.endswith(".npz"))[0]
    q = np.load(os.path.join(C.BOX_DATA_DIR, name))["qpos"]
    return q[0, 39:43].astype(np.float64)                        # box freejoint quat


def carton_frame():
    """The carton as (M, centre, dims): M's columns are (side, side, UP) unit axes in the
    scan's local frame, right-handed, with UP the axis the clips' rest quaternion sends to
    world +Z. dims are the box's extents along those same axes."""
    R, ctr, ext = source_obb()
    qr = rest_quat()
    world = np.array([quat.mul_vec(qr[None, :], R[:, i][None, :])[0] for i in range(3)])
    iz = int(np.argmax(np.abs(world[:, 2])))                     # axis that stands up at rest
    up = R[:, iz] * np.sign(world[iz, 2])                        # ...pointing UP, not down
    ia = (iz + 1) % 3
    e0 = R[:, ia]
    e1 = np.cross(up, e0)                                        # right-handed by construction
    M = np.stack([e0, e1, up], axis=1)
    dims = np.array([ext[ia], ext[3 - iz - ia], ext[iz]])        # extents along e0, e1, up
    return M, ctr, dims


def face_quads(dims):
    """The six outward quads of an axis-aligned box of size `dims` centred at the origin, as
    (name, base, uvec, vvec). Corners are base + i*uvec + j*vvec for i,j in {0,1}, ordered so
    cross(uvec, vvec) is the outward normal. Each face's v runs 'up' the print."""
    hx, hy, hz = dims / 2.0
    return [
        ("top",    (-hx, -hy,  hz), (2 * hx, 0, 0), (0, 2 * hy, 0)),   # +Z: u=+X, v=+Y
        ("bottom", ( hx, -hy, -hz), (-2 * hx, 0, 0), (0, 2 * hy, 0)),  # -Z: u=-X, v=+Y
        ("+x",     ( hx, -hy, -hz), (0, 2 * hy, 0), (0, 0, 2 * hz)),   # +X: u=+Y, v=+Z
        ("-x",     (-hx,  hy, -hz), (0, -2 * hy, 0), (0, 0, 2 * hz)),  # -X: u=-Y, v=+Z
        ("+y",     ( hx,  hy, -hz), (-2 * hx, 0, 0), (0, 0, 2 * hz)),  # +Y: u=-X, v=+Z
        ("-y",     (-hx, -hy, -hz), (2 * hx, 0, 0), (0, 0, 2 * hz)),   # -Y: u=+X, v=+Z
    ]


# --------------------------------------------------------------------------
# Texture
# --------------------------------------------------------------------------
def kraft_panel(rng):
    """One 512px cardboard face: fibre noise + corrugation ribs + edge wear."""
    n = CELL
    fibre = rng.normal(0, 7.0, (n, n, 1))                       # paper grain
    blot = rng.normal(0, 3.0, (n // 16, n // 16))               # slow mottling
    blot = np.array(Image.fromarray(np.clip(blot + 128, 0, 255).astype(np.uint8))
                    .resize((n, n), Image.BICUBIC), float)[..., None] - 128.0
    ribs = 3.5 * np.sin(np.arange(n) * 2 * np.pi / 9.0)[None, :, None]   # corrugation
    img = np.array(KRAFT, float) + fibre + 2.0 * blot + ribs

    ii = np.minimum(np.arange(n), n - 1 - np.arange(n))
    d = np.minimum(ii[:, None], ii[None, :]).astype(float)      # px distance to nearest edge
    img *= (0.78 + 0.22 * np.clip(d / 26.0, 0, 1))[..., None]   # bevelled, worn edges
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def draw_cross(dr, cx, cy, size, width, colour):
    h, w = size / 2.0, width / 2.0
    dr.rectangle([cx - h, cy - w, cx + h, cy + w], fill=colour)
    dr.rectangle([cx - w, cy - h, cx + w, cy + h], fill=colour)


def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


def centred(dr, cy, text, f, colour):
    x0, y0, x1, y1 = dr.textbbox((0, 0), text, font=f)
    dr.text(((CELL - (x1 - x0)) / 2 - x0, cy - (y1 - y0) / 2 - y0), text, font=f, fill=colour)


def draw_closure(img, rng, tape=True):
    """Turn a bare kraft panel into a CLOSED lid: the two major flaps meet at a seam across
    the middle, sealed with a strip of packing tape. Drawn straight into the pixel array so
    the tape can be a translucent, faintly glossy overlay rather than a flat rectangle."""
    a = np.array(img, float)
    n = CELL
    y = np.arange(n)[:, None]
    mid = n / 2.0

    # the flap seam: a dark hairline where the two flap edges butt together, with the far flap
    # casting a soft shadow onto the near one
    seam = np.exp(-((y - mid) ** 2) / (2 * 1.6 ** 2))            # the join itself
    shade = np.exp(-np.clip(y - mid, 0, None) / 7.0) * (y > mid)  # shadow on the lower flap
    a *= (1.0 - 0.55 * seam - 0.16 * shade)[..., None]

    if tape:
        half = 34
        band = (np.abs(y - mid) < half).astype(float)
        # translucent amber tape. Keep the lift small and warm-tinted: the lid is the face the
        # overhead light hits square-on, so a brighter strip reads as a white band, not tape.
        tint = np.array([1.06, 0.99, 0.86])                      # amber, slightly darker than kraft
        edge = np.exp(-((np.abs(y - mid) - half) ** 2) / (2 * 3.0 ** 2)) * band
        gloss = np.exp(-((y - mid + 11) ** 2) / (2 * 7.0 ** 2)) * band
        wrinkle = 1.0 + 0.02 * np.sin(np.arange(n)[None, :] / 6.0 + rng.normal(0, 0.4))
        a = a * (1 - band * 0.7) + band * 0.7 * (a * tint) * wrinkle[..., None]
        a *= (1.0 - 0.26 * edge)[..., None]                      # tape edges catch dirt
        a += (16 * gloss)[..., None]                             # narrow specular streak
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def panel_top(rng):
    """Lid: a taped flap closure, with the print sitting on the near flap."""
    img = draw_closure(kraft_panel(rng), rng)
    dr = ImageDraw.Draw(img)
    centred(dr, 120, "MEDICINE", font("DejaVuSans-Bold.ttf", 70), INK)
    dr.line([100, 172, CELL - 100, 172], fill=INK, width=3)
    centred(dr, 205, "HANDLE WITH CARE", font("DejaVuSans-Bold.ttf", 26), INK)
    centred(dr, 400, "THIS SIDE UP", font("DejaVuSans-Bold.ttf", 30), INK)
    return img


def panel_side(rng):
    img = kraft_panel(rng)
    dr = ImageDraw.Draw(img)
    centred(dr, 215, "MEDICINE", font("DejaVuSans-Bold.ttf", 72), INK)
    dr.line([100, 268, CELL - 100, 268], fill=INK, width=3)
    draw_cross(dr, CELL / 2, 330, 62, 20, RED)
    centred(dr, 400, "HANDLE WITH CARE", font("DejaVuSans-Bold.ttf", 24), INK)
    return img


def panel_bottom(rng):
    """Underside: the same taped flap closure, unprinted."""
    return draw_closure(kraft_panel(rng), rng)


def build_texture(path, seed=7):
    rng = np.random.default_rng(seed)
    atlas = Image.new("RGB", (COLS * CELL, ROWS * CELL), KRAFT)
    panels = {"top": panel_top, "bottom": panel_bottom, "+x": panel_side,
              "-x": panel_side, "+y": panel_side, "-y": panel_side}
    for name, (col, row) in FACE_CELLS.items():
        atlas.paste(panels[name](rng), (col * CELL, row * CELL))
    atlas.save(path)
    return atlas.size


# --------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------
def build_obj(path, M, ctr, dims, atlas_wh):
    """Write the UV-mapped carton. Each face gets its own 4 verts, so the v/vt indices coincide
    and MuJoCo needs no re-indexing to attach the texcoords. Vertices are built axis-aligned
    then mapped into the scan's local frame by (M, ctr) -- M is a rotation (det +1), so the
    outward winding survives."""
    aw, ah = atlas_wh
    lines = ["# Generated by tools/make_medicine_box.py -- do not hand-edit.",
             "# 'MEDICINE' carton, built in largebox.obj's own (tilted) box frame.",
             "# No mtllib: the texture is bound by the MuJoCo material in scene_box.xml.",
             "o medicinebox"]
    verts, uvs, faces = [], [], []
    for name, base, uvec, vvec in face_quads(dims):
        col, row = FACE_CELLS[name]
        u0 = (col * CELL + INSET) / aw
        u1 = ((col + 1) * CELL - INSET) / aw
        # image y grows downward, OBJ v grows upward -- flip so face-v points up the print
        v0 = 1.0 - ((row + 1) * CELL - INSET) / ah
        v1 = 1.0 - (row * CELL + INSET) / ah
        n = len(verts)
        for i, j in ((0, 0), (1, 0), (1, 1), (0, 1)):
            p = np.array([base[k] + i * uvec[k] + j * vvec[k] for k in range(3)])
            verts.append(ctr + M @ p)
            uvs.append((u0 + i * (u1 - u0), v0 + j * (v1 - v0)))
        faces += [(n + 1, n + 2, n + 3), (n + 1, n + 3, n + 4)]      # OBJ is 1-based
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in uvs]
    lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in faces]
    open(path, "w").write("\n".join(lines) + "\n")
    return len(verts), len(faces)


def main():
    M, ctr, dims = carton_frame()
    png = os.path.join(OUT_DIR, "medicinebox.png")
    obj = os.path.join(OUT_DIR, "medicinebox.obj")
    wh = build_texture(png)
    nv, nf = build_obj(obj, M, ctr, dims, wh)
    print(f"texture {png}  {wh[0]}x{wh[1]}")
    print(f"mesh    {obj}  {nv} verts, {nf} tris")
    print(f"carton  {np.round(dims, 4)} m (w x d x h), centre {np.round(ctr, 4)} in scan frame")
    print(f"up axis {np.round(M[:, 2], 3)} of the scan's local frame")


if __name__ == "__main__":
    main()
