#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
import cython


ctypedef fused image_t:
    cython.uchar[:, :]
    cython.double[:, :]


cpdef moments_central(image_t image, double cy, double cx, Py_ssize_t order):
    cdef Py_ssize_t p, q, r, cols
    cdef double y, x, dy, dx, dxp, dyq
    cdef double[:, ::1] mu = np.zeros((order + 1, order + 1), dtype=np.double)
    cols = image.shape[1]
    if cols == 2:
        for r in range(image.shape[0]):
            y = image[r][0]
            x = image[r][1]
            dy = y - cy
            dx = x - cx
            dxp = 1
            for p in range(order + 1):
                dyq = 1
                for q in range(order + 1):
                    mu[p, q] += dyq * dxp
                    dyq *= dy
                dxp *= dx
        return np.asarray(mu)

    cdef Py_ssize_t c
    for r in range(image.shape[0]):
        dy = r - cy
        for c in range(cols):
            dx = c - cx
            y = image[r, c]
            dxp = 1
            for p in range(order + 1):
                dyq = 1
                for q in range(order + 1):
                    mu[p, q] += y * dyq * dxp
                    dyq *= dy
                dxp *= dx
    return np.asarray(mu)


def moments_hu(double[:, :] nu):
    cdef double[::1] hu = np.zeros((7, ), dtype=np.double)
    cdef double t0 = nu[3, 0] + nu[1, 2]
    cdef double t1 = nu[2, 1] + nu[0, 3]
    cdef double q0 = t0 * t0
    cdef double q1 = t1 * t1
    cdef double n4 = 4 * nu[1, 1]
    cdef double s = nu[2, 0] + nu[0, 2]
    cdef double d = nu[2, 0] - nu[0, 2]
    hu[0] = s
    hu[1] = d * d + n4 * nu[1, 1]
    hu[3] = q0 + q1
    hu[5] = d * (q0 - q1) + n4 * t0 * t1
    t0 *= q0 - 3 * q1
    t1 *= 3 * q0 - q1
    q0 = nu[3, 0] - 3 * nu[1, 2]
    q1 = 3 * nu[2, 1] - nu[0, 3]
    hu[2] = q0 * q0 + q1 * q1
    hu[4] = q0 * t0 + q1 * t1
    hu[6] = q1 * t0 - q0 * t1
    return np.asarray(hu)
