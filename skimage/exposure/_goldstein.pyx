#cython: cdivision=True
#cython: boundscheck=True
#cython: nonecheck=False
#cython: wraparound=True

from __future__ import print_function

import numpy as np

from libc.math cimport M_PI, lround
cimport numpy as cnp


cdef inline double _phase_difference(double from_, double to):
    cdef double d = to - from_
    if d > M_PI:
        d -= 2 * M_PI
    elif d < -M_PI:
        d += 2 * M_PI
    return d


def find_phase_residues_cy(double[:, ::1] image):
    residues_array = np.zeros((image.shape[0], image.shape[1]),
                              dtype=np.int8, order='C')
    cdef:
        cnp.int8_t[:, ::1] residues = residues_array
        Py_ssize_t i, j
        double s
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            s = (_phase_difference(image[i - 1, j - 1], image[i - 1, j])
                 + _phase_difference(image[i - 1, j], image[i, j])
                 + _phase_difference(image[i, j], image[i, j - 1])
                 + _phase_difference(image[i, j - 1], image[i - 1, j - 1]))
            residues[i, j] = lround(s / (2 * M_PI))
    return residues_array
