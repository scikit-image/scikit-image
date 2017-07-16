#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
import cython


ctypedef fused image_t:
    cython.uchar[:, :]
    cython.double[:, :]


cpdef moments_central(image_t image, double cr, double cc, Py_ssize_t order):
    cdef Py_ssize_t p, q, r, c
    cdef double val, dr, dc, dcp, drq
    cdef double[:, ::1] mu = np.zeros((order + 1, order + 1), dtype=np.double)

    for r in range(image.shape[0]):
        dr = r - cr
        for c in range(image.shape[1]):
            dc = c - cc
            val = image[r, c]
            dcp = 1
            for p in range(order + 1):
                drq = 1
                for q in range(order + 1):
                    mu[p, q] += val * drq * dcp
                    drq *= dr
                dcp *= dc
    return np.asarray(mu)


cpdef moments_contour_central(image_t contour, double cy, double cx, Py_ssize_t order):
    cdef Py_ssize_t p, q, r, cols
    cdef double y, x, dy, dx, dxp, dyq
    cdef double[:, ::1] mu = np.zeros((order + 1, order + 1), dtype=np.double)
    
    for r in range(contour.shape[0]):
        y = contour[r][0]
        x = contour[r][1]
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
