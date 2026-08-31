"""GenoView-style feature database with a smoothed simulation root, built from our library.

Math + layout mirror ../GenoViewPython-MotionMatching/genoview_g1.py's build_database:
a per-clip Savitzky-Golay-smoothed "simulation root" (ground position + facing) carries the
character, with the pelvis stored as a local offset of it; the 27-D search feature is then
expressed in that smoothed root frame:
  Xpos (6)  local foot positions (L,R) relative to the sim root
  Xvel (9)  local velocities of both feet + the pelvis
  XtrajPos (6) future sim-root xy at +10/+20/+30 frames, in the sim-heading frame
  XtrajDir (6) future sim heading xy at the same horizons
Normalized by a per-block scale (one shared std per block) so the search weights blocks sensibly.
"""
import numpy as np
from scipy.signal import savgol_filter

from . import config as C
from . import quat
from .states import Skill

FPS = C.FPS
HORIZONS = np.array(C.HORIZONS)
FORWARD = np.array([1.0, 0.0, 0.0])     # G1 pelvis forward axis
UP = np.array([0.0, 0.0, 1.0])


def yaw_quat(theta):
    """Quaternion (wxyz) for a rotation of theta about world +Z."""
    return quat.from_angle_axis(np.asarray(theta), UP)


def heading_dir(rootquat):
    """World-space forward direction (xy, z=0, normalized) of a root quaternion (wxyz)."""
    fwd = quat.mul_vec(rootquat, FORWARD) * np.array([1.0, 1.0, 0.0])
    return fwd / (np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-9)


def smooth_root(pelvisPos_world, headDir):
    """Build the smoothed simulation root for one clip range.
    Returns (simPos (N,3) ground, simTheta (N,), headDirSmooth (N,3))."""
    n = len(pelvisPos_world)
    pw = min(C.ROOT_POS_SMOOTH, n if n % 2 == 1 else n - 1)
    dw = min(C.ROOT_DIR_SMOOTH, n if n % 2 == 1 else n - 1)
    simXY = pelvisPos_world[:, :2]
    if pw >= 5:
        simXY = savgol_filter(simXY, pw, 3, axis=0, mode='interp')
    simPos = np.concatenate([simXY, np.zeros((n, 1))], axis=1)
    d = headDir[:, :2].copy()
    if dw >= 5:
        d = savgol_filter(d, dw, 3, axis=0, mode='interp')
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    headDirSmooth = np.concatenate([d, np.zeros((n, 1))], axis=1)
    simTheta = np.arctan2(d[:, 1], d[:, 0])
    return simPos, simTheta, headDirSmooth


def central_diff(x, fps):
    """Central-difference velocity along axis 0 with linear endpoint extrapolation."""
    v = np.empty_like(x)
    if len(x) < 4:
        v[:] = (np.gradient(x, axis=0) * fps) if len(x) > 1 else 0.0
        return v
    v[1:-1] = 0.5 * (x[2:] - x[1:-1]) * fps + 0.5 * (x[1:-1] - x[:-2]) * fps
    v[0] = v[1] - (v[3] - v[2])
    v[-1] = v[-2] + (v[-2] - v[-3])
    return v


def central_diff_ang(rot, fps):
    """Angular velocity (scaled-angle-axis) from a quaternion series via central differences."""
    n = len(rot)
    ang = np.zeros((n, 3))
    if n < 4:
        if n >= 2:
            ang[1:] = quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rot[1:], rot[:-1]))) * fps
            ang[0] = ang[1]
        return ang
    ang[1:-1] = (0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rot[2:], rot[1:-1]))) * fps +
                 0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rot[1:-1], rot[:-2]))) * fps)
    ang[0] = ang[1] - (ang[3] - ang[2])
    ang[-1] = ang[-2] + (ang[-2] - ang[-3])
    return ang


def build_db(lib):
    """Assemble the GenoView (smoothed-sim-root) feature DB (a dict) from the G1 library."""
    qpos = lib["qpos"].astype(np.float64)
    fic = lib["frame_in_clip"]
    starts = np.where(fic == 0)[0]
    stops = np.append(starts[1:], len(qpos))
    spans = list(zip(starts, stops))

    rootQuat = qpos[:, 3:7].copy()
    dof = qpos[:, 7:].copy()
    footL = lib["feet_world"][:, 0].astype(np.float64)   # world foot positions
    footR = lib["feet_world"][:, 1].astype(np.float64)
    pelvis = qpos[:, 0:3]                                 # floating base == pelvis
    headDirRaw = heading_dir(rootQuat)                   # (T,3)
    boxPosW = lib["box_pose"][:, 0:3].astype(np.float64)  # world box position
    boxRotW = lib["box_pose"][:, 3:7].astype(np.float64)  # world box quaternion (wxyz)

    # ---- Smoothed simulation root + pelvis-local offset, per range ----
    T = len(qpos)
    simPos = np.zeros((T, 3))
    simTheta = np.zeros(T)
    headDir = np.zeros((T, 3))                            # smoothed heading
    pelvLocalPos = np.zeros((T, 3))
    pelvLocalRot = np.zeros((T, 4))
    for rs, re in spans:
        sp, st, hd = smooth_root(pelvis[rs:re], headDirRaw[rs:re])
        simPos[rs:re], simTheta[rs:re], headDir[rs:re] = sp, st, hd
        qh = yaw_quat(st)
        pelvLocalPos[rs:re] = quat.inv_mul_vec(qh, pelvis[rs:re] - sp)
        pelvLocalRot[rs:re] = quat.mul(quat.inv(qh), rootQuat[rs:re])

    # ---- Per-range central-difference velocities (never cross range seams) ----
    def clipwise_vel(arr):
        v = np.zeros_like(arr)
        for rs, re in spans:
            v[rs:re] = central_diff(arr[rs:re], FPS)
        return v

    footLvel, footRvel = clipwise_vel(footL), clipwise_vel(footR)
    pelvisVel, simVel = clipwise_vel(pelvis), clipwise_vel(simPos)
    dofVel, pelvLocalVel = clipwise_vel(dof), clipwise_vel(pelvLocalPos)
    boxVelW = clipwise_vel(boxPosW)                       # world box linear velocity
    yawRate = np.zeros(T)
    pelvLocalAng = np.zeros((T, 3))
    for rs, re in spans:
        yawRate[rs:re] = central_diff(np.unwrap(simTheta[rs:re])[:, None], FPS)[:, 0]
        pelvLocalAng[rs:re] = central_diff_ang(pelvLocalRot[rs:re], FPS)

    # ---- Pose + trajectory features in the smoothed sim-root frame ----
    qh_all = yaw_quat(simTheta)
    to_local = lambda v: quat.inv_mul_vec(qh_all, v)

    Xpos = np.concatenate([to_local(footL - simPos), to_local(footR - simPos)], -1)        # (T,6)
    Xvel = np.concatenate([to_local(footLvel), to_local(footRvel), to_local(pelvisVel)], -1)  # (T,9)
    XtrajPos = np.zeros((T, 6))
    XtrajDir = np.zeros((T, 6))
    for rs, re in spans:
        idx = np.arange(rs, re)
        for k, h in enumerate(HORIZONS):
            ft = np.clip(idx + h, rs, re - 1)
            XtrajPos[rs:re, 2 * k:2 * k + 2] = quat.inv_mul_vec(
                qh_all[rs:re], simPos[ft] - simPos[rs:re])[:, 0:2]
            XtrajDir[rs:re, 2 * k:2 * k + 2] = quat.inv_mul_vec(qh_all[rs:re], headDir[ft])[:, 0:2]

    # ---- Box features in the same sim-root (gravity-aligned base) frame ----
    # Stored relative to the smoothed root exactly like the pelvis, so reconstructing the
    # box at runtime (root o boxLocal) rides along with the character. boxLocalRot is also
    # kept as a scaled-angle-axis for the search query (orientation block). boxLocalPosVel /
    # boxLocalAng are the time-derivatives of the *local* pose (mirroring pelvLocalVel /
    # pelvLocalAng) -- the velocity terms the box inertialization matches at a cut.
    boxLocalPos = quat.inv_mul_vec(qh_all, boxPosW - simPos)               # (T,3)
    boxLocalRot = quat.mul(quat.inv(qh_all), boxRotW)                      # (T,4)
    boxLocalVel = quat.inv_mul_vec(qh_all, boxVelW)                        # (T,3) world vel (query)
    boxLocalAA = quat.to_scaled_angle_axis(quat.abs(boxLocalRot))          # (T,3)
    boxLocalPosVel = clipwise_vel(boxLocalPos)                             # (T,3) d/dt(local pos)
    boxLocalAng = np.zeros((T, 3))
    for rs, re in spans:
        boxLocalAng[rs:re] = central_diff_ang(boxLocalRot[rs:re], FPS)     # (T,3) local ang. vel

    # ---- Three normalized search databases (genoview-style per-block scaling) ----
    # Each frame belongs to one skill; we normalize each database over its own frames so the
    # block statistics match what is actually searched. A block's scale is its (shared) std
    # divided by an optional weight, so a heavier block contributes more to the L2 distance.
    skill = lib["skill"] if "skill" in lib else np.zeros(T, np.int32)
    masks = {s.db: skill == s for s in
             (Skill.LOCO, Skill.CARRY, Skill.PICK, Skill.PLACE)}

    def make_db(blocks, mask):
        """blocks: list of (array (T,d), weight). Returns (Xn, offset, scale) over mask."""
        if not mask.any():                            # no frames of this skill in the library
            mask = np.ones(T, bool)
        X = np.concatenate([b for b, _ in blocks], -1)
        offset = X[mask].mean(0)
        scale = np.concatenate([np.repeat(b[mask].std(0).mean() / w, b.shape[1])
                                for b, w in blocks])
        scale = np.where(scale < 1e-5, 1.0, scale)
        return ((X - offset) / scale).astype(np.float32), offset, scale

    Wr, Wv = C.BOX_ROT_WEIGHT, C.BOX_VEL_WEIGHT
    pose = [(Xpos, 1.0), (Xvel, 1.0)]
    traj = [(XtrajPos, 1.0), (XtrajDir, 1.0)]
    box = lambda wp, wr=Wr: [(boxLocalPos, wp), (boxLocalAA, wr), (boxLocalVel, Wv)]
    # PICK and PLACE share the 24-D (pose + box, no trajectory) feature space but are SEPARATE
    # databases, so pick can weight the box position AND orientation more
    # (PICK_BOX_POS_WEIGHT / PICK_BOX_ROT_WEIGHT) than place -- the entry is chosen mainly by
    # where the box sits and which way it faces in the base frame.
    dbs = {
        "loco": make_db(pose + traj, masks["loco"]),                        # 27-D (unchanged)
        "carry": make_db(pose + traj + box(C.BOX_POS_WEIGHT), masks["carry"]),   # 36-D
        "pick": make_db(pose + box(C.PICK_BOX_POS_WEIGHT,
                                   C.PICK_BOX_ROT_WEIGHT), masks["pick"]),       # 24-D
        "place": make_db(pose + box(C.BOX_POS_WEIGHT), masks["place"]),          # 24-D
    }
    dbs = {k: dict(X=Xn, offset=off, scale=sc) for k, (Xn, off, sc) in dbs.items()}

    return dict(
        starts=starts, stops=stops, spans=spans,
        dof=dof, dofVel=dofVel,
        simPos=simPos, simTheta=simTheta, simVel=simVel, yawRate=yawRate,
        pelvLocalPos=pelvLocalPos, pelvLocalVel=pelvLocalVel,
        pelvLocalRot=pelvLocalRot, pelvLocalAng=pelvLocalAng,
        boxLocalPos=boxLocalPos, boxLocalRot=boxLocalRot,
        boxLocalPosVel=boxLocalPosVel, boxLocalAng=boxLocalAng,
        # raw (un-normalized) blocks so the controller can assemble a cross-database query
        # (pose from the current frame, trajectory from the command, box from the live box).
        rawXpos=Xpos, rawXvel=Xvel, rawTrajPos=XtrajPos, rawTrajDir=XtrajDir,
        rawBoxPos=boxLocalPos, rawBoxAA=boxLocalAA, rawBoxVel=boxLocalVel,
        dbs=dbs,
        # back-compat aliases: the locomotion database is the default "X".
        X=dbs["loco"]["X"], Xoffset=dbs["loco"]["offset"], Xscale=dbs["loco"]["scale"])
