# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False

from __future__ import division

from libc.float cimport DBL_MAX
from libc.math cimport sqrt

import numpy as np

cdef int type1 = 1
cdef int type2 = 2
cdef int type3 = 3

cdef inline double euclidean(double x, double y):
    return sqrt((x - y) * (x - y))

cdef inline double min3(double[3] v):
    cdef int i, m = 0

    for i in range(1, 3):
        if v[i] < v[m]:
            m = i

    return v[m]


def dtw(double[:] x, double[:] y, int case=1, int start_anchor_slack=0,
        int end_anchor_slack=0):
    """Return mapping between two curves based on dynamic time warping.

    DTW is an algorithm for measuring similarity between two sequences which
    may vary in time or speed

    For instance, similarities in walking patterns would be detected, even if
    in one video the person was walking slowly and if in another he or she were
    walking more quickly, or even if there were accelerations and decelerations
    during the course of one observation.

    Anchoring sets an upper limit on the number of elements that may be
    excluded in either sequence when mapping.

    Parameters
    ----------
    x, y : 1D array, dtype float64
        Input sequences
    case : int, {1, 2, 3}
        Type-1 DTW uses 27-, 45- and 63-degree local path constraint.
        Type-2 DTW uses 0-, 45- and 90-degree local path constraint.
        Type-3 DTW uses a combination of Type-1 and Type-2
    start_anchor_slack : int
        Maximum deviation allowed from start boundary condition.
    end_anchor_slack : int
        Maximum deviation allowed from end boundary condition.

    References
    ----------
    .. [1] http://mirlab.org/jang/books/dcpr/dpDtw.asp?title=8-4%20Dynamic%20
    Time%20Warping
    .. [2] http://en.wikipedia.org/wiki/Dynamic_time_warping

    """
    cdef:
        int m = len(x)
        int n = len(y)
        double[:, ::1] distance
        Py_ssize_t i, j, min_i, min_j
        double[3] costs

    if len(x) > 2 * len(y) or len(y) > 2 * len(x):
        raise ValueError("Sequence lengths cannot differ by more than 50%.")

    distance = np.zeros((m + 2, n + 2)) + DBL_MAX
    distance[1, 1] = 0

    # Populate distance matrix
    for i in range(2, 3+start_anchor_slack):
        distance[i, 2] = euclidean(x[i - 2], y[0])

    for j in range(2, 3+start_anchor_slack):
        distance[2, j] = euclidean(x[0], y[j-2])

    for i in range(3, m + 2):
        for j in range(3, n + 2):
            costs[0] = distance[i - 1, j - 1]
            costs[1] = distance[i - 1, j]
            costs[2] = distance[i, j - 1]

            distance[i, j] = euclidean(x[i - 2], y[j - 2]) + min3(costs)

    # Trace back
    cdef list path = []

    i = m + 1
    j = n + 1

    for c in range(end_anchor_slack):
        if distance[i - 1, n + 1] < distance[i, j]:
            i = i - 1
            j = n + 1

    for c in range(end_anchor_slack):
        if distance[m + 1, j - 1] < distance[i, j]:
            i = m + 1
            j = j - 1

    while i > 2 and j > 2:
        path.append((i - 2, j - 2))

        min_i, min_j = i - 1, j - 1

        if case == type1 or case == type3:
            if distance[i - 2, j - 1] < distance[min_i, min_j]:
                min_i, min_j = i - 2, j - 1

            if distance[i - 1, j - 2] < distance[min_i, min_j]:
                min_i, min_j = i - 1, j - 2

        if case == type2 or case == type3:
            if distance[i, j - 1] < distance[min_i, min_j]:
                min_i, min_j = i, j - 1

            if distance[i - 1, j] < distance[min_i, min_j]:
                min_i, min_j = i - 1, j

        i, j = min_i, min_j

    path.append((i - 2, j - 2))

    return path, np.asarray(distance)
