"""Quaternion helpers matching the SceneBot web demo's JS math exactly.

The demo works in xyzw internally (gl-matrix style); clip data and MuJoCo
qpos are wxyz. Function names note the convention they expect.
"""
import numpy as np


def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], float)


def xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], float)


def quat_mul_xyzw(a, b):
    # the JS (gG) normalizes both inputs before multiplying
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na > 0:
        a = a / na
    if nb > 0:
        b = b / nb
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quat_conj_xyzw(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], float)


def quat_rotate_xyzw(q, v):
    x, y, z, w = np.asarray(q, float)
    vx, vy, vz = np.asarray(v, float)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ])


def yaw_quat_xyzw(yaw):
    h = 0.5 * yaw
    return np.array([0.0, 0.0, np.sin(h), np.cos(h)])


def yaw_from_wxyz(q):
    w, x, y, z = np.asarray(q, float)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_to_mat33_xyzw(q):
    x, y, z, w = np.asarray(q, float)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ])
