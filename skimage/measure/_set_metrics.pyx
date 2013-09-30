#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from libc.math cimport sqrt

import numpy as np


MAX_FLOAT64 = np.finfo(np.float64).max


def hausdorff_distance_onesided(cnp.float64_t[:, ::1] points_sup,
                                cnp.float64_t[:, ::1] points_inf):
    """
    Compute the one-sided Haussdorff distance between two sets of points.
    """
    assert points_sup.shape[1] == points_inf.shape[1]

    cdef double d2_max = 0.
    cdef double d2_min_i = MAX_FLOAT64
    cdef double d2_j = 0.
    cdef Py_ssize_t i, j, k

    for i in range(points_sup.shape[0]):
        d2_min_i = MAX_FLOAT64
        for j in range(points_inf.shape[0]):
            d2_j = 0.
            for k in range(points_sup.shape[1]):
                d2_j += (points_sup[i, k] - points_inf[j, k])**2
            d2_min_i = d2_min_i if d2_min_i < d2_j else d2_j
        d2_max = d2_max if d2_max > d2_min_i else d2_min_i

    return sqrt(d2_max)
