#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np

cimport numpy as cnp


def moments(double[:, :] image, Py_ssize_t order=3):
    return central_moments(image, 0, 0, order)


def central_moments(double[:, :] image, double cr, double cc,
                    Py_ssize_t order=3):
    cdef Py_ssize_t p, q, r, c
    cdef double[:, ::1] mu = np.zeros((order + 1, order + 1), dtype=np.double)
    for p in range(order + 1):
        for q in range(order + 1):
            for r in range(image.shape[0]):
                for c in range(image.shape[1]):
                    mu[p, q] += image[r, c] * (r - cr) ** q * (c - cc) ** p
    return np.asarray(mu)


def normalized_moments(double[:, :] mu, Py_ssize_t order=3):
    cdef Py_ssize_t p, q
    cdef double[:, ::1] nu = np.zeros((order + 1, order + 1), dtype=np.double)
    for p in range(order + 1):
        for q in range(order + 1):
            if p + q >= 2:
                nu[p,q] = mu[p, q] / mu[0, 0] ** (<double>(p + q) / 2 + 1)
            else:
                nu[p,q] = np.nan
    return np.asarray(nu)


def hu_moments(double[:, :] nu):
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
    q0 = nu[3, 0]- 3 * nu[1, 2]
    q1 = 3 * nu[2, 1] - nu[0, 3]
    hu[2] = q0 * q0 + q1 * q1
    hu[4] = q0 * t0 + q1 * t1
    hu[6] = q1 * t0 - q0 * t1
    return np.asarray(hu)
