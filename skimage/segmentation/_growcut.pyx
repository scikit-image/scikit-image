#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from __future__ import division

import numpy as np
from skimage import io, img_as_float

cimport cython
cimport numpy as cnp
from cython.parallel import parallel, prange

cdef extern from "math.h" nogil:
    double sqrt(double)


cdef inline double distance(double[:, :, ::1] image,
                            Py_ssize_t r0, Py_ssize_t c0,
                            Py_ssize_t r1, Py_ssize_t c1) nogil:
    cdef:
        double s = 0, d
        int i

    for i in range(3):
        d = image[r0, c0, i] - image[r1, c1, i]
        s += d * d

    return sqrt(s)


cdef double s3 = sqrt(3)


cdef inline double g(double d) nogil:
    return 1 - (d / s3)


def growcut(image, state,
            int max_iter=500, int window_size=5):
    """Grow-cut segmentation.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    state : (M, N, 2) ndarray
        Initial state, which stores (foreground/background, strength) for
        each pixel position or automaton.  The strength represents the
        certainty of the state (e.g., 1 is a hard seed value that remains
        constant throughout segmentation).
    max_iter : int, optional
        The maximum number of automata iterations to allow.  The segmentation
        may complete earlier if the state no longer varies.
    window_size : int
        Size of the neighborhood window.

    Returns
    -------
    mask : ndarray
        Segmented image.  A value of zero indicates background, one foreground.

    Reference
    ---------
    .. [1] V. Vezhnevets, V. Konouchine. "Grow-Cut - Interactive Multi-Label
           N-D Image Segmentation".  In Proceedings of the 2005 Conference,
           Graphicon. Pages 150-156.

    """
    cdef:
        double[:, :, ::1] state_arr, state_next_arr, image_arr

        Py_ssize_t i, j, ii, jj, width, height, ws, n, changes, changes_per_cell
        double[:] C_p, S_p, C_q, S_q
        double gc, attack_strength, defense_strength, winning_colony

    image_arr = np.ascontiguousarray(img_as_float(image))
    state_arr = state

    height, width = image.shape[0], image.shape[1]
    ws = (window_size - 1) // 2

    changes = 1
    changes_per_cell = 0
    n = 0

    state_next_arr = state_arr.copy()

    while changes > 0 and n < max_iter:
        changes = 0
        n += 1

        for j in range(width):
            for i in range(height):

                winning_colony = state_arr[i, j, 0]
                defense_strength = state_arr[i, j, 1]

                for jj in xrange(max(0, j - ws), min(j + ws + 1, width)):
                    for ii in xrange(max(0, i - ws), min(i + ws + 1, height)):
                        if ii == i and jj == j:
                            continue

                        # p -> current cell, (i, j)
                        # q -> attacker, (ii, jj)

                        gc = g(distance(image_arr, i, j, ii, jj))

                        attack_strength = gc * state_arr[ii, jj, 1]

                        if attack_strength > defense_strength:
                            defense_strength = attack_strength
                            winning_colony = state_arr[ii, jj, 0]
                            changes += 1

                state_next_arr[i, j, 0] = winning_colony
                state_next_arr[i, j, 1] = defense_strength

        state_next_arr, state_arr = state_arr, state_next_arr

    return np.asarray(state_next_arr[:, :, 0])
