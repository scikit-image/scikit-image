#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import os
import numpy as np

from .. import data_dir

cimport numpy as cnp
from libc.math cimport sin, cos

from .._shared.interpolation cimport round

POS = np.loadtxt(os.path.join(data_dir, "orb_descriptor_positions.txt"),
                 dtype=np.int8)
POS0 = np.ascontiguousarray(POS[:, :2])
POS1 = np.ascontiguousarray(POS[:, 2:])


def _orb_loop(double[:, ::1] image, Py_ssize_t[:, ::1] keypoints,
              double[:] orientations):

    cdef Py_ssize_t i, d, kr, kc, pr0, pr1, pc0, pc1, spr0, spc0, spr1, spc1
    cdef int[:, ::1] steered_pos0, steered_pos1
    cdef double angle
    cdef unsigned char[:, ::1] descriptors = \
        np.zeros((keypoints.shape[0], POS.shape[0]), dtype=np.uint8)
    cdef signed char[:, ::1] cpos0 = POS0
    cdef signed char[:, ::1] cpos1 = POS1

    for i in range(descriptors.shape[0]):

        angle = orientations[i]
        sin_a = sin(angle)
        cos_a = cos(angle)

        kr = keypoints[i, 0]
        kc = keypoints[i, 1]

        for j in range(descriptors.shape[1]):
            pr0 = cpos0[j, 0]
            pc0 = cpos0[j, 1]
            pr1 = cpos1[j, 0]
            pc1 = cpos1[j, 1]

            spr0 = <Py_ssize_t>round(sin_a * pr0 + cos_a * pc0)
            spc0 = <Py_ssize_t>round(cos_a * pr0 - sin_a * pc0)
            spr1 = <Py_ssize_t>round(sin_a * pr1 + cos_a * pc1)
            spc1 = <Py_ssize_t>round(cos_a * pr1 - sin_a * pc1)

            if image[kr + spr0, kc + spc0] < image[kr + spr1, kc + spc1]:
                descriptors[i, j] = True

    return np.asarray(descriptors)
