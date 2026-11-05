"""Port of lmm-g1-box's web/js/math.js (vectors [x, y, z], quaternions [w, x, y, z], Z-up, +X forward).
The springs and inertializers port spring.h of orangeduck/Motion-Matching. Plain floats and lists: these
run on single 3- and 4-vectors many times per step, where numpy's call overhead costs more than the math."""
import math

LN2 = math.log(2.0)


def hypot(*a):
    """Math.hypot of V8 (builtins/math.tq): scaled by the largest value, Kahan-summed. Python's math.hypot rounds
    differently in the last bit, and the page has decisions on the sign of a difference of such values."""
    m = 0.0
    for v in a:
        v = abs(v)
        if v > m:
            m = v
    if m == 0.0 or m == math.inf:
        return m
    s = comp = 0.0
    for v in a:
        n = abs(v) / m
        summand = n * n - comp
        pre = s + summand
        comp = (pre - s) - summand
        s = pre
    return math.sqrt(s) * m


def add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def scale(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def length(a):
    return hypot(a[0], a[1], a[2])


def lerp(a, b, t):
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]


def norm(a, eps=1e-8):
    l = hypot(a[0], a[1], a[2]) + eps
    return [a[0] / l, a[1] / l, a[2] / l]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def qid():
    return [1.0, 0.0, 0.0, 0.0]


def qmul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def qinv(a):
    return [a[0], -a[1], -a[2], -a[3]]


def qnorm(a):
    l = hypot(a[0], a[1], a[2], a[3])
    return [a[0] / l, a[1] / l, a[2] / l, a[3] / l]


def qabs(a):
    return [-a[0], -a[1], -a[2], -a[3]] if a[0] < 0 else a


def qmul_vec(a, v):
    w, x, y, z = a
    tx = 2 * (y * v[2] - z * v[1])
    ty = 2 * (z * v[0] - x * v[2])
    tz = 2 * (x * v[1] - y * v[0])
    return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)]


def qinv_mul_vec(a, v):
    return qmul_vec(qinv(a), v)


def qmul_inv(a, b):
    return qmul(a, qinv(b))


def qinv_mul(a, b):
    return qmul(qinv(a), b)


def qfrom_angle_axis(angle, axis):
    s = math.sin(angle / 2)
    return [math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s]


def qfrom_yaw(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def qyaw(a):
    f = qmul_vec(a, [1.0, 0.0, 0.0])
    return math.atan2(f[1], f[0])


def qto_scaled_angle_axis(a, eps=1e-8):
    l = hypot(a[1], a[2], a[3])
    half = math.atan2(l, a[0])
    s = (2 * half) / (l + eps)
    return [a[1] * s, a[2] * s, a[3] * s]


def qfrom_scaled_angle_axis(v, eps=1e-8):
    angle = hypot(v[0], v[1], v[2])
    if angle < eps:
        return [1.0, v[0] / 2, v[1] / 2, v[2] / 2]
    return qfrom_angle_axis(angle, [v[0] / angle, v[1] / angle, v[2] / angle])


def qslerp_shortest_approx(a, b, t):
    d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    bb = [-b[0], -b[1], -b[2], -b[3]] if d < 0 else b
    return qnorm([a[0] + (bb[0] - a[0]) * t, a[1] + (bb[1] - a[1]) * t,
                  a[2] + (bb[2] - a[2]) * t, a[3] + (bb[3] - a[3]) * t])


def qangle_between(a, b):
    d = qabs(qmul_inv(a, b))
    return 2 * math.acos(min(1.0, abs(d[0])))


def qfrom_matrix(c0, c1, c2):
    m00, m10, m20 = c0
    m01, m11, m21 = c1
    m02, m12, m22 = c2
    if m22 < 0:
        r = ([m21 - m12, 1 + m00 - m11 - m22, m10 + m01, m02 + m20] if m00 > m11
             else [m02 - m20, m10 + m01, 1 - m00 + m11 - m22, m21 + m12])
    else:
        r = ([m10 - m01, m02 + m20, m21 + m12, 1 - m00 - m11 + m22] if m00 < -m11
             else [1 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01])
    return qnorm(r)


def qfrom_xform_xy(t):
    """Two matrix columns [c0x, c1x, c0y, c1y, c0z, c1z] (row-major 3x2) -> quaternion."""
    c0 = [t[0], t[2], t[4]]
    c1 = [t[1], t[3], t[5]]
    c2 = norm(cross(c0, c1))
    c1 = norm(cross(c2, c0))
    c0 = norm(c0)
    return qfrom_matrix(c0, c1, c2)


def halflife_to_damping(h, eps=1e-5):
    return (4 * LN2) / (h + eps)


def damp_factor(h, dt, eps=1e-5):
    return 1 - math.exp(-(LN2 * dt) / (h + eps))


def simple_spring_damper_quat(x, v, goal, halflife, dt):
    """Returns the new (x, v)."""
    y = halflife_to_damping(halflife) / 2
    j0 = qto_scaled_angle_axis(qabs(qmul(x, qinv(goal))))
    j1 = add(v, scale(j0, y))
    e = math.exp(-y * dt)
    return (qmul(qfrom_scaled_angle_axis(scale(add(j0, scale(j1, dt)), e)), goal),
            scale(sub(v, scale(j1, y * dt)), e))


def decay_spring_vec(x, v, halflife, dt):
    y = halflife_to_damping(halflife) / 2
    j1 = add(v, scale(x, y))
    e = math.exp(-y * dt)
    return scale(add(x, scale(j1, dt)), e), scale(sub(v, scale(j1, y * dt)), e)


def decay_spring_quat(x, v, halflife, dt):
    y = halflife_to_damping(halflife) / 2
    j0 = qto_scaled_angle_axis(x)
    j1 = add(v, scale(j0, y))
    e = math.exp(-y * dt)
    return qfrom_scaled_angle_axis(scale(add(j0, scale(j1, dt)), e)), scale(sub(v, scale(j1, y * dt)), e)


def decay_spring_scalar(x, v, halflife, dt):
    """x and v may be numpy arrays: the same arithmetic per element as the JS loop over the joints."""
    y = halflife_to_damping(halflife) / 2
    j1 = v + x * y
    e = math.exp(-y * dt)
    return e * (x + j1 * dt), e * (v - j1 * y * dt)


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def js_sign1(v):
    """Math.sign(v || 1): 1 for 0."""
    return 1.0 if v == 0 or v > 0 else -1.0
