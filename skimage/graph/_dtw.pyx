# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False

from __future__ import division

from libc.float cimport DBL_MAX
from libc.math cimport fabs

import numpy as np

cdef int TYPE1 = 1
cdef int TYPE2 = 2
cdef int TYPE3 = 3


cdef inline double min3(double[3] v):
    cdef double m = v[0]

    if v[1] < m:
        m = v[1]
    if v[2] < m:
        m = v[2]
    return m


def dtw(double[:] x, double[:] y, int case=1, int start_anchor_slack=0,
        int end_anchor_slack=0, distance=None):
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
        Type-3 DTW uses a combination of Type-1 and Type-2.
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
        double[:, ::1] _distance
        Py_ssize_t i, j, min_i, min_j
        double[3] costs

    if len(x) > 2 * len(y) or len(y) > 2 * len(x):
        raise ValueError("Sequence lengths cannot differ by more than 50%.")

    if distance == None:
        _distance = np.zeros((m + 2, n + 2)) + DBL_MAX
        _distance[1, 1] = 0

        # Populate distance matrix
        for i in range(2, 3+start_anchor_slack):
            _distance[i, 2] = fabs(x[i - 2] - y[0])

        for j in range(2, 3+start_anchor_slack):
            _distance[2, j] = fabs(x[0] - (y[j - 2]))

        for i in range(3, m + 2):
            for j in range(3, n + 2):
                costs[0] = _distance[i - 1, j - 1]
                costs[1] = _distance[i - 1, j]
                costs[2] = _distance[i, j - 1]

                _distance[i, j] = fabs(x[i - 2] - y[j - 2]) + min3(costs)
    else:
        _distance = distance

    # Trace back
    cdef list path = []

    i = m + 1
    j = n + 1

    for c in range(end_anchor_slack):
        if _distance[i - 1, n + 1] < _distance[i, j]:
            i = i - 1
            j = n + 1

    for c in range(end_anchor_slack):
        if _distance[m + 1, j - 1] < _distance[i, j]:
            i = m + 1
            j = j - 1

    while i > 2 and j > 2:
        path.append((i - 2, j - 2))

        min_i, min_j = i - 1, j - 1

        if case == TYPE1 or case == TYPE3:
            if _distance[i - 2, j - 1] < _distance[min_i, min_j]:
                min_i, min_j = i - 2, j - 1

            if _distance[i - 1, j - 2] < _distance[min_i, min_j]:
                min_i, min_j = i - 1, j - 2

        if case == TYPE2 or case == TYPE3:
            if _distance[i, j - 1] < _distance[min_i, min_j]:
                min_i, min_j = i, j - 1

            if _distance[i - 1, j] < _distance[min_i, min_j]:
                min_i, min_j = i - 1, j

        i, j = min_i, min_j

    path.append((i - 2, j - 2))

    return path, np.asarray(_distance[3:, 3:])
