#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from libc.math cimport sqrt
from libc.float cimport DBL_MAX
from scipy.spatial import cKDTree

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

    tree = cKDTree(points_inf)
    return max(tree.query(points_sup, k=1)[0])
