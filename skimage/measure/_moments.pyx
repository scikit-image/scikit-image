#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
import numpy as np
cimport numpy as np


def central_moments(np.ndarray[np.double_t, ndim=2] array, double cr, double cc,
                     int order):
    cdef int p, q, r, c
    cdef np.ndarray[np.double_t, ndim=2] mu
    mu = np.zeros((order + 1, order + 1), 'double')
    for p in range(order + 1):
        for q in range(order + 1):
            for r in range(array.shape[0]):
                for c in range(array.shape[1]):
                    mu[p,q] += array[r,c] * (r - cr) ** q * (c - cc) ** p
    return mu

def normalized_moments(np.ndarray[np.double_t, ndim=2] mu, int order):
    cdef int p, q
    cdef np.ndarray[np.double_t, ndim=2] nu
    nu = np.zeros((order + 1, order + 1), 'double')
    for p in range(order + 1):
        for q in range(order + 1):
            if p + q >= 2:
                nu[p,q] = mu[p,q] / mu[0,0]**(<double>(p + q) / 2 + 1)
            else:
                nu[p,q] = np.nan
    return nu

def hu_moments(np.ndarray[np.double_t, ndim=2] nu):
    cdef np.ndarray[np.double_t, ndim=1] hu = np.zeros((7,), 'double')
    cdef double t0 = nu[3,0] + nu[1,2]
    cdef double t1 = nu[2,1] + nu[0,3]
    cdef double q0 = t0 * t0
    cdef double q1 = t1 * t1
    cdef double n4 = 4 * nu[1,1]
    cdef double s = nu[2,0] + nu[0,2]
    cdef double d = nu[2,0] - nu[0,2]
    hu[0] = s
    hu[1] = d * d + n4 * nu[1,1]
    hu[3] = q0 + q1
    hu[5] = d * (q0 - q1) + n4 * t0 * t1
    t0 *= q0 - 3 * q1
    t1 *= 3 * q0 - q1
    q0 = nu[3,0]- 3 * nu[1,2]
    q1 = 3 * nu[2,1] - nu[0,3]
    hu[2] = q0 * q0 + q1 * q1
    hu[4] = q0 * t0 + q1 * t1
    hu[6] = q1 * t0 - q0 * t1
    return hu
