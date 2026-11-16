"""Port of the part of lmm-g1-box's web/js/edit/mirror.js that pickmotion.js reaches: row-major 3x3 to quaternion
and back, and the vertical plane the pick editor mirrors in.

Left out: jointMirror, Plane.mirrorQpos, towardMirrorQpos and towardMirrorPose, which only the symmetry and the
check of pickedit.js use, and the pick motion never turns symmetry on.
"""
import math

from .mathq import qfrom_matrix, qfrom_yaw, qinv, qmul


def mat_quat(R):
    return qfrom_matrix([R[0], R[3], R[6]], [R[1], R[4], R[7]], [R[2], R[5], R[8]])


def quat_mat(a):
    w, x, y, z = a
    return [
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]


def mirror_local_quat(a):
    """The mirror image of a rotation in a frame whose xz-plane is the mirror."""
    return [a[0], -a[1], a[2], -a[3]]


class Plane:
    """The vertical plane through c = [x, y] that holds the direction of `yaw`. Its frame has x along that
    direction, y to the left of it and z up."""

    def __init__(self, c, yaw):
        self.c = [c[0], c[1]]
        self.yaw = yaw
        self.cos = math.cos(yaw)
        self.sin = math.sin(yaw)
        self.q = qfrom_yaw(yaw)

    def local(self, p):
        dx, dy = p[0] - self.c[0], p[1] - self.c[1]
        return [self.cos * dx + self.sin * dy, -self.sin * dx + self.cos * dy, p[2] if len(p) > 2 else 0.0]

    def world(self, l):
        return [self.c[0] + self.cos * l[0] - self.sin * l[1],
                self.c[1] + self.sin * l[0] + self.cos * l[1], l[2] if len(l) > 2 else 0.0]

    def dir(self, l):
        return [self.cos * l[0] - self.sin * l[1], self.sin * l[0] + self.cos * l[1]]

    def mirror_point(self, p):
        l = self.local(p)
        return self.world([l[0], -l[1], l[2]])

    def mirror_quat(self, a):
        return qmul(self.q, mirror_local_quat(qmul(qinv(self.q), a)))
