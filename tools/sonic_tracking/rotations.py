"""wxyz quaternion helpers, ported 1:1 from the C++ deploy stack's
math_utils.hpp (which itself mirrors the IsaacGym/poselib Python functions).
All quaternions are (w, x, y, z); vectorized over leading axes where noted.
"""

import numpy as np


def quat_mul(a, b):
    """Hamilton product a*b, both (…,4) wxyz."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_conjugate(q):
    out = np.array(q, dtype=float, copy=True)
    out[..., 1:] *= -1.0
    return out


def quat_rotate(q, v):
    """Rotate vector(s) v (…,3) by quaternion(s) q (…,4)."""
    q = np.asarray(q, float)
    v = np.asarray(v, float)
    qw = q[..., 0:1]
    qv = q[..., 1:]
    a = v * (2.0 * qw * qw - 1.0)
    b = np.cross(qv, v) * qw * 2.0
    c = qv * np.sum(qv * v, axis=-1, keepdims=True) * 2.0
    return a + b + c


def calc_heading(q):
    """Yaw of the rotated x-axis (matches calc_heading_d)."""
    rot_dir = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
    return np.arctan2(rot_dir[..., 1], rot_dir[..., 0])


def quat_from_angle_axis(angle, axis):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * angle
    return np.concatenate([np.atleast_1d(np.cos(half)),
                           np.sin(half) * axis])


def calc_heading_quat(q):
    return quat_from_angle_axis(calc_heading(q), np.array([0.0, 0.0, 1.0]))


def calc_heading_quat_inv(q):
    return quat_from_angle_axis(-calc_heading(q), np.array([0.0, 0.0, 1.0]))


def euler_z_to_quat(yaw):
    return quat_from_angle_axis(yaw, np.array([0.0, 0.0, 1.0]))


def quat_to_mat(q):
    """3x3 rotation matrix from a single wxyz quaternion (normalized first)."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_6d(q):
    """First two COLUMNS of R, flattened row-wise -> 6 values (the
    motion_anchor_ori format: R00,R01,R10,R11,R20,R21)."""
    R = quat_to_mat(q)
    return R[:, :2].reshape(-1)


def quat_slerp(q0, q1, t):
    """Spherical interpolation between two wxyz quaternions (scalar t)."""
    q0 = np.asarray(q0, float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, float) / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def quat_to_angle_axis(q):
    """(angle, axis) of a single wxyz quaternion, angle in [-pi, pi]."""
    q = np.asarray(q, float) / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    sin_half = np.linalg.norm(q[1:])
    angle = 2.0 * np.arctan2(sin_half, q[0])
    axis = q[1:] / sin_half if sin_half > 1e-12 else np.array([1.0, 0.0, 0.0])
    return angle, axis
