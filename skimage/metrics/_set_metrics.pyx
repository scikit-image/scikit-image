#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from libc.math cimport sqrt
from libc.float cimport DBL_MAX

import numpy as np

def hausdorff_distance_onesided(cnp.float64_t[:, ::1] points_sup,
                                cnp.float64_t[:, ::1] points_inf):
    """
    Compute the one-sided Hausdorff distance between two sets of points.

    The Hausdorff one-sided distance is the maximum distance between any
    point in ``points_sup`` and its nearest point on ``points_inf``.

    Parameters
    ----------
    points_sup, points_inf : (N, M) ndarray of float
        Array containing the coordinates of ``N`` points in an ``M``
        dimensional space.

    Returns
    -------
    distance : float
        The Hausdorff one-sided distance between sets ``points_sup`` and
        ``points_inf``, using Euclidean distance to calculate the distance
        between points in ``points_sup`` and ``points_inf``.

    """
    assert points_sup.shape[1] == points_inf.shape[1]

    cdef double d2_max = 0.
    cdef double d2_min_i = DBL_MAX
    cdef double d2_j = 0.
    cdef Py_ssize_t i, j, k

    for i in range(points_sup.shape[0]):
        d2_min_i = DBL_MAX
        for j in range(points_inf.shape[0]):
            d2_j = 0.
            for k in range(points_sup.shape[1]):
                d2_j += (points_sup[i, k] - points_inf[j, k])**2
            d2_min_i = d2_min_i if d2_min_i < d2_j else d2_j
            if d2_j < d2_max:
                break
        d2_max = d2_max if d2_max > d2_min_i else d2_min_i

    return sqrt(d2_max)
