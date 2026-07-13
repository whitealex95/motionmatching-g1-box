#!/usr/bin/env python3
"""Export the G1 motion-matching database + model skeleton for the browser demo (docs/).

Run from the repo root with the mujoco env, e.g.:
    ~/miniconda3/envs/deploy_mujoco/bin/python tools/export_web_data.py

Writes:
  docs/data/model.json  -- kinematic tree (bodies: parent, local pos/quat, joint axis/qadr)
                           used by the JS forward-kinematics + skeleton renderer.
  docs/data/mesh.{json,bin} -- G1 visual meshes (body-local), placed by the JS FK.
  docs/data/boxmesh.{json,bin} -- the interactive box mesh (body-local), placed by box_qpos.
  docs/data/mm.json     -- header: per-array {dtype, shape, byte offset} into mm.bin, plus
                           clip metadata, jump + pick/place entries, carry segments, the loco
                           search-clip indices, and the box config the JS state machine needs.
  docs/data/mm.bin      -- all the runtime arrays the JS matcher needs, concatenated.

The JS matcher (docs/js/mm.js) is a 1:1 port of mm_g1/controller.py + features.build_db,
so we export exactly the per-skill databases + box arrays build_db() produces. A self-check
verifies our pure-numpy FK (the same formula the JS uses) matches MuJoCo before writing.
"""
import os
import sys
import json
import numpy as np
import mujoco
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from mm_g1 import config as C
from mm_g1.data import load_library
from mm_g1.features import build_db
from mm_g1.jumps import jump_entries
from mm_g1 import boxes
from mm_g1 import quat

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "docs", "data")


# --------------------------------------------------------------------------
# Model kinematic tree (for the JS forward-kinematics skeleton renderer)
# --------------------------------------------------------------------------
def export_model():
    m = mujoco.MjModel.from_xml_path(C.SCENE_XML)
    bodies = []
    for b in range(1, m.nbody):                      # skip world (0)
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        # the single hinge joint of this body, if any (free joint -> root, axis=None)
        axis, qadr = None, -1
        for j in range(m.njnt):
            if m.jnt_bodyid[j] == b and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                axis = [float(x) for x in m.jnt_axis[j]]
                qadr = int(m.jnt_qposadr[j])
                break
        bodies.append(dict(
            name=name,
            parent=int(m.body_parentid[b]) - 1,      # -1 == root (its parent is world)
            pos=[float(x) for x in m.body_pos[b]],
            quat=[float(x) for x in m.body_quat[b]],  # wxyz
            axis=axis, qadr=qadr))
    return m, bodies


def _fk_numpy(bodies, qpos):
    """The exact FK the JS renderer runs: world pos/quat per body from a (36,) qpos.
    Root (body 0) uses qpos[0:7]; each child applies body offset then its hinge rotation."""
    n = len(bodies)
    wp = np.zeros((n, 3)); wq = np.zeros((n, 4))
    wp[0], wq[0] = qpos[0:3], qpos[3:7]              # root body == pelvis
    for i in range(1, n):
        b = bodies[i]
        p = b["parent"]
        lp, lq = np.array(b["pos"]), np.array(b["quat"])
        wp[i] = wp[p] + quat.mul_vec(wq[p], lp)
        r = quat.mul(wq[p], lq)
        if b["axis"] is not None:
            r = quat.mul(r, quat.from_angle_axis(qpos[b["qadr"]], np.array(b["axis"])))
        wq[i] = r
    return wp, wq


def export_meshes(m):
    """Extract every visual mesh from the compiled model into body-local space and write
    docs/data/mesh.{json,bin}. Positions are float32; indices uint16 (per-geom, 0-based);
    normals are recomputed in JS (flat shading), so we don't ship them. Each geom records
    its body index + rgba so the JS renderer can colour and FK-place it."""
    geoms, pos_chunks, idx_chunks = [], [], []
    vbase, ibase = 0, 0     # running vertex count, running uint16-index count
    for g in range(m.ngeom):
        if m.geom_group[g] != 2 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(m.geom_dataid[g])
        va, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        fa, nf = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        verts = m.mesh_vert[va:va + nv].astype(np.float64)   # (nv,3), mesh-local frame
        faces = m.mesh_face[fa:fa + nf].astype(np.int64)     # (nf,3), 0-based within the mesh
        assert nv < 65536, f"mesh {mid} has {nv} verts (>uint16)"
        # mesh -> body frame: v_body = geom_pos + R(geom_quat) * v_mesh
        gp, gq = m.geom_pos[g], m.geom_quat[g]
        vb = gp + quat.mul_vec(np.tile(gq, (nv, 1)), verts)
        rgba = (m.mat_rgba[int(m.geom_matid[g])] if m.geom_matid[g] >= 0 else m.geom_rgba[g])
        geoms.append(dict(body=int(m.geom_bodyid[g]) - 1, vstart=vbase, vcount=nv,
                          istart=ibase, icount=nf * 3, rgba=[float(c) for c in rgba[:3]]))
        pos_chunks.append(vb.astype(np.float32))
        idx_chunks.append(faces.astype(np.uint16))
        vbase += nv
        ibase += nf * 3

    positions = np.concatenate(pos_chunks).ravel()           # float32 (vbase*3,)
    indices = np.concatenate(idx_chunks).ravel()             # uint16  (ibase,)
    blob = positions.tobytes() + indices.tobytes()
    meta = dict(nverts=int(vbase), nidx=int(ibase), idx_byte_offset=positions.nbytes, geoms=geoms)
    json.dump(meta, open(os.path.join(OUT, "mesh.json"), "w"))
    with open(os.path.join(OUT, "mesh.bin"), "wb") as f:
        f.write(blob)
    print(f"  mesh.json + mesh.bin: {len(geoms)} geoms, {vbase} verts, "
          f"{ibase // 3} tris, {len(blob) / 1e6:.1f} MB")


def export_box_texture(m, matid):
    """Write the box material's texture to docs/data/boxmesh.png, straight out of the compiled
    model -- so the browser gets exactly the pixels MuJoCo renders, with no asset path to keep
    in sync. Returns True if the material carries one."""
    texids = [int(t) for t in np.atleast_1d(m.mat_texid[matid]) if t >= 0]
    if not texids:
        return False
    t = texids[0]
    w, h, nc, adr = (int(m.tex_width[t]), int(m.tex_height[t]),
                     int(m.tex_nchannel[t]), int(m.tex_adr[t]))
    px = m.tex_data[adr:adr + w * h * nc].reshape(h, w, nc)
    Image.fromarray(px[:, :, :3]).save(os.path.join(OUT, "boxmesh.png"))
    print(f"  boxmesh.png: {w}x{h}")
    return True


def export_box_mesh():
    """Extract the interactive box's visual mesh (from the G1+box scene) into body-local
    space and write docs/data/boxmesh.{json,bin} (+ boxmesh.png if the box is textured). The
    box is a free body driven every frame by the matcher's box_qpos, so the JS renderer places
    this one mesh group directly from that pose -- it is NOT part of the FK body tree
    (model.json)."""
    m = mujoco.MjModel.from_xml_path(C.SCENE_BOX_XML)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "largebox")
    pos_chunks, uv_chunks, idx_chunks, rgba = [], [], [], [0.82, 0.52, 0.22]
    vbase, textured = 0, False
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(m.geom_dataid[g])
        va, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        fa, nf = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        verts = m.mesh_vert[va:va + nv].astype(np.float64)
        faces = m.mesh_face[fa:fa + nf].astype(np.int64)
        assert nv < 65536, f"box mesh has {nv} verts (>uint16)"
        gp, gq = m.geom_pos[g], m.geom_quat[g]                 # mesh -> body frame
        vb = gp + quat.mul_vec(np.tile(gq, (nv, 1)), verts)
        pos_chunks.append(vb.astype(np.float32))
        idx_chunks.append((faces + vbase).astype(np.uint16))

        # Texcoords. MuJoCo stores v in IMAGE space (v=0 at the top row), the flip of the OBJ
        # convention it loaded them from; three.js (texture.flipY defaults to true) wants the
        # OBJ convention back, so undo the flip here or the print comes out upside down.
        ta, tn = int(m.mesh_texcoordadr[mid]), int(m.mesh_texcoordnum[mid])
        if ta >= 0 and tn == nv:
            uv = m.mesh_texcoord[ta:ta + tn].astype(np.float32).copy()
            uv[:, 1] = 1.0 - uv[:, 1]
            uv_chunks.append(uv)

        matid = int(m.geom_matid[g])
        if matid >= 0:
            rgba = [float(c) for c in m.mat_rgba[matid][:3]]
            textured |= export_box_texture(m, matid)
        else:
            rgba = [float(c) for c in m.geom_rgba[g][:3]]
        vbase += nv

    positions = np.concatenate(pos_chunks).ravel()
    indices = np.concatenate(idx_chunks).ravel()
    has_uv = textured and len(uv_chunks) and sum(len(u) for u in uv_chunks) == vbase
    uvs = np.concatenate(uv_chunks).ravel() if has_uv else np.zeros(0, np.float32)

    blob = positions.tobytes() + uvs.tobytes() + indices.tobytes()
    meta = dict(nverts=int(vbase), nidx=int(len(indices)), rgba=rgba,
                uv_byte_offset=(positions.nbytes if has_uv else -1),
                idx_byte_offset=positions.nbytes + uvs.nbytes,
                texture=("boxmesh.png" if has_uv else None))
    json.dump(meta, open(os.path.join(OUT, "boxmesh.json"), "w"))
    with open(os.path.join(OUT, "boxmesh.bin"), "wb") as f:
        f.write(blob)
    print(f"  boxmesh.json + boxmesh.bin: {vbase} verts, {len(indices) // 3} tris, "
          f"{'uv-mapped, ' if has_uv else ''}{len(blob) / 1e6:.1f} MB")


def verify_fk(m, bodies, n_tests=5):
    """Confirm our pure-numpy FK matches MuJoCo's, so the JS port renders correctly."""
    data = mujoco.MjData(m)
    rng = np.random.RandomState(0)
    worst = 0.0
    for _ in range(n_tests):
        q = np.zeros(m.nq)
        q[3:7] = [1, 0, 0, 0]
        q[0:3] = rng.uniform(-1, 1, 3)
        q[3:7] = quat.normalize(rng.uniform(-1, 1, 4))
        q[7:] = rng.uniform(-1, 1, m.nq - 7)
        data.qpos[:] = q
        mujoco.mj_kinematics(m, data)
        wp, _ = _fk_numpy(bodies, q)
        # body i in our list == model body i+1 (we skipped world)
        worst = max(worst, float(np.abs(wp - data.xpos[1:]).max()))
    print(f"  FK self-check vs MuJoCo: worst body position error = {worst:.2e} m")
    assert worst < 1e-6, "JS FK formula would not match MuJoCo!"


# --------------------------------------------------------------------------
# Motion-matching database (everything the JS matcher reads)
# --------------------------------------------------------------------------
def export_mm(lib):
    db = build_db(lib)
    jump_enter, jump_land_of = jump_entries(lib)
    starts, stops = db["starts"], db["stops"]
    skill = lib["skill"]
    box_attach = lib["box_attach"]
    T = len(db["dof"])
    H = int(max(C.HORIZONS))

    # Per-skill feature databases (the JS matcher queries each with its own offset/scale).
    dbs = db["dbs"]
    Xloco, Xcarry = dbs["loco"]["X"], dbs["carry"]["X"]

    # Loco search clips = pure-locomotion clips (skill all 0) long enough for a full horizon.
    search_clips = [int(ci) for ci, (rs, re) in enumerate(zip(starts, stops))
                    if not skill[rs:re].any() and re - rs > H]
    # Contiguous CARRY segments (searched like locomotion, among carry frames only).
    carry_segs = np.asarray(boxes.carry_segments(lib), np.int32).reshape(-1, 2)

    # Pick / place ENTRY frames + the phase-end frame each ride finishes at. We ship the
    # entry-frame rows of the pick/place databases (the only rows ever queried) so the JS can
    # nearest-neighbour the live pose+box query to them.
    pick_enter, pick_end_of, place_enter, place_end_of = boxes.box_entries(lib)
    pick_end = np.asarray([pick_end_of[int(f)] for f in pick_enter], np.int32)
    place_end = np.asarray([place_end_of[int(f)] for f in place_enter], np.int32)
    pickEnterX = dbs["pick"]["X"][pick_enter] if len(pick_enter) else np.zeros((0, 24), np.float32)
    placeEnterX = dbs["place"]["X"][place_enter] if len(place_enter) else np.zeros((0, 24), np.float32)

    # Box resting orientation for the interactive spawn (box-in-base at a representative pick
    # entry == its world orientation when the robot faces +x); mirrors MotionMatcher.
    box_spawn_rot = (db["boxLocalRot"][int(pick_enter[0])].copy()
                     if len(pick_enter) else np.array([1.0, 0, 0, 0]))

    arrays = {
        # per-skill normalized feature matrices + their query offsets/scales
        "Xloco": Xloco.astype(np.float32),
        "Xcarry": Xcarry.astype(np.float32),
        "locoOffset": dbs["loco"]["offset"].astype(np.float32),
        "locoScale": dbs["loco"]["scale"].astype(np.float32),
        "carryOffset": dbs["carry"]["offset"].astype(np.float32),
        "carryScale": dbs["carry"]["scale"].astype(np.float32),
        "pickOffset": dbs["pick"]["offset"].astype(np.float32),
        "pickScale": dbs["pick"]["scale"].astype(np.float32),
        "placeOffset": dbs["place"]["offset"].astype(np.float32),
        "placeScale": dbs["place"]["scale"].astype(np.float32),
        # raw (un-normalized) pose blocks the query is assembled from
        "rawXpos": db["rawXpos"].astype(np.float32),
        "rawXvel": db["rawXvel"].astype(np.float32),
        # pose reconstruction + inertialization
        "dof": db["dof"].astype(np.float32),
        "dofVel": db["dofVel"].astype(np.float32),
        "simPos": db["simPos"].astype(np.float32),
        "simTheta": db["simTheta"].astype(np.float32),
        "simVel": db["simVel"].astype(np.float32),
        "yawRate": db["yawRate"].astype(np.float32),
        "pelvLocalPos": db["pelvLocalPos"].astype(np.float32),
        "pelvLocalVel": db["pelvLocalVel"].astype(np.float32),
        "pelvLocalRot": db["pelvLocalRot"].astype(np.float32),
        "pelvLocalAng": db["pelvLocalAng"].astype(np.float32),
        # box reconstruction + inertialization (box-in-base, like the pelvis)
        "boxLocalPos": db["boxLocalPos"].astype(np.float32),
        "boxLocalRot": db["boxLocalRot"].astype(np.float32),
        "boxLocalPosVel": db["boxLocalPosVel"].astype(np.float32),
        "boxLocalAng": db["boxLocalAng"].astype(np.float32),
        "box_attach": box_attach.astype(np.int32),
        # clip bookkeeping + skill labels
        "starts": starts.astype(np.int32),
        "stops": stops.astype(np.int32),
        "clip_id": lib["clip_id"].astype(np.int32),
        "frame_in_clip": lib["frame_in_clip"].astype(np.int32),
        "lengths": lib["lengths"].astype(np.int32),
        "skill": skill.astype(np.int32),
        # skill entries / segments
        "jump_enter": np.asarray(jump_enter, np.int32),
        "jump_land": np.asarray([jump_land_of[int(f)] for f in jump_enter], np.int32),
        "search_clips": np.asarray(search_clips, np.int32),
        "carry_segs": carry_segs,
        "pick_enter": np.asarray(pick_enter, np.int32),
        "pick_end": pick_end,
        "place_enter": np.asarray(place_enter, np.int32),
        "place_end": place_end,
        "pickEnterX": pickEnterX.astype(np.float32),
        "placeEnterX": placeEnterX.astype(np.float32),
        "box_spawn_rot": box_spawn_rot.astype(np.float32),
    }

    blob = bytearray()
    header = {}
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        header[name] = dict(dtype=a.dtype.name, shape=list(a.shape), offset=len(blob))
        blob += a.tobytes()

    meta = dict(
        fps=C.FPS, ndof=29, horizons=list(map(int, C.HORIZONS)),
        max_speed=C.MAX_SPEED, walk_scale=C.WALK_SCALE, carry_max_speed=C.CARRY_MAX_SPEED,
        search_time=C.SEARCH_TIME, current_bias=C.CURRENT_BIAS,
        inert_halflife=C.INERT_HALFLIFE, vel_halflife=C.VEL_HALFLIFE,
        rot_halflife=C.ROT_HALFLIFE, box_inert_halflife=C.BOX_INERT_HALFLIFE,
        phase_touchdown=C.PHASE_TOUCHDOWN, phase_after=C.PHASE_AFTER,
        pick_radius=C.PICK_RADIUS, box_spawn_fwd=C.BOX_SPAWN_FWD,
        box_spawn_lat=C.BOX_SPAWN_LAT, box_rest_z=C.BOX_REST_Z,
        skill_loco=C.SKILL_LOCO, skill_pick=C.SKILL_PICK,
        skill_carry=C.SKILL_CARRY, skill_place=C.SKILL_PLACE,
        clip_names=[str(n) for n in lib["clip_names"]],
        arrays=header, n_frames=T)
    return meta, bytes(blob)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Exporting G1 web demo data -> docs/data/")
    m, bodies = export_model()
    verify_fk(m, bodies)
    json.dump(dict(bodies=bodies, nbody=len(bodies)),
              open(os.path.join(OUT, "model.json"), "w"))
    print(f"  model.json: {len(bodies)} bodies")
    export_meshes(m)
    export_box_mesh()

    lib = load_library()
    meta, blob = export_mm(lib)
    json.dump(meta, open(os.path.join(OUT, "mm.json"), "w"))
    with open(os.path.join(OUT, "mm.bin"), "wb") as f:
        f.write(blob)
    print(f"  mm.json + mm.bin: {meta['n_frames']} frames, "
          f"{len(meta['clip_names'])} clips, {len(blob) / 1e6:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
