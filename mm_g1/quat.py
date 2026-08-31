import numpy as np

def _fast_cross(a, b):
    o = np.empty(np.broadcast(a, b).shape)
    o[...,0] = a[...,1]*b[...,2] - a[...,2]*b[...,1]
    o[...,1] = a[...,2]*b[...,0] - a[...,0]*b[...,2]
    o[...,2] = a[...,0]*b[...,1] - a[...,1]*b[...,0]
    return o

def eye(shape, dtype=np.float32):
    return np.ones(list(shape) + [4], dtype=dtype) * np.asarray([1, 0, 0, 0], dtype=dtype)

def length(x):
    return np.sqrt(np.sum(x * x, axis=-1))

def normalize(x, eps=1e-8):
    return x / (length(x)[...,np.newaxis] + eps)

def abs(x):
    return np.where(x[...,0:1] > 0.0, x, -x)

def from_angle_axis(angle, axis):
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    q = np.concatenate([c, s * axis], axis=-1)
    return q

def inv(q):
    return np.asarray([1, -1, -1, -1], dtype=np.float32) * q

def mul(x, y):
    x0, x1, x2, x3 = x[...,0], x[...,1], x[...,2], x[...,3]
    y0, y1, y2, y3 = y[...,0], y[...,1], y[...,2], y[...,3]
    
    o = np.empty(np.broadcast(x, y).shape)
    o[...,0] = y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3
    o[...,1] = y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2
    o[...,2] = y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1
    o[...,3] = y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0
    return o

def mul_inv(x, y):
    return mul(x, inv(y))

def mul_vec(q, x):
    t = 2.0 * _fast_cross(q[...,1:], x)
    return x + q[...,0][...,None] * t + _fast_cross(q[..., 1:], t)

def inv_mul_vec(q, x):
    return mul_vec(inv(q), x)

def between(x, y):
    return np.concatenate([
        np.sqrt(np.sum(x*x, axis=-1) * np.sum(y*y, axis=-1))[...,np.newaxis] + 
        np.sum(x * y, axis=-1)[...,np.newaxis], 
        _fast_cross(x, y)], axis=-1)
        
def log(x, eps=1e-5):
    length = np.sqrt(np.sum(np.square(x[...,1:]), axis=-1))[...,np.newaxis]
    safe = np.where(length < eps, np.ones_like(length), length)   # avoid 0/0 in the masked branch
    halfangle = np.where(length < eps, np.ones_like(length), np.arctan2(length, x[...,0:1]) / safe)
    return halfangle * x[...,1:]
    
def exp(x, eps=1e-5):
    halfangle = np.sqrt(np.sum(np.square(x), axis=-1))[...,np.newaxis]
    c = np.where(halfangle < eps, np.ones_like(halfangle), np.cos(halfangle))
    s = np.where(halfangle < eps, np.ones_like(halfangle), np.sinc(halfangle / np.pi))
    return np.concatenate([c, s * x], axis=-1)
    
def to_scaled_angle_axis(x, eps=1e-5):
    return 2.0 * log(x, eps)
    
def from_scaled_angle_axis(x, eps=1e-5):
    return exp(x / 2.0, eps)
