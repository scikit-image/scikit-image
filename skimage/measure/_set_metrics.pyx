#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from libc.math cimport sqrt

import numpy as np

def hausdorff_distance_onesided(cnp.float64_t[:, ::1] points_sup,
                                cnp.float64_t[:, ::1] points_inf):
    """
    Compute the one-sided Haussdorff distance between two sets of points.
    """
    assert points_sup.shape[1] == points_inf.shape[1]

    cdef double d_h2 = 0.
    cdef double d2 = 99999999999.
    cdef double acc = 0.
    cdef Py_ssize_t i, j, k

    for i in range(points_sup.shape[0]):
        d2 = 99999999999.
        for j in range(points_inf.shape[0]):
            acc = 0.
            for k in range(points_sup.shape[1]):
                acc += (points_sup[i, k] - points_inf[j, k])**2
            d2 = d2 if d2 < acc else acc
        d_h2 = d_h2 if d_h2 > d2 else d2

    return sqrt(d_h2)
