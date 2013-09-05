#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp

import numpy as np


def _orb_loop(double[:, ::1] image, char[:, ::1] descriptors, Py_ssize_t[:, ::1] keypoints,
              double[:] orientations, int[:, ::1] pos0, int[:, ::1] pos1):

    cdef Py_ssize_t i, d, kr, kc, pr0, pr1, pc0, pc1
    cdef int[:, ::1] steered_pos0, steered_pos1

    for i in range(keypoints.shape[0]):
        angle = orientations[i]
        a = np.sin(angle * np.pi / 180.)
        b = np.cos(angle * np.pi / 180.)
        rotation_matrix = np.asarray([[b, a], [-a, b]])
        steered_pos0 = np.dot(pos0, rotation_matrix).astype(np.int32)
        steered_pos1 = np.dot(pos1, rotation_matrix).astype(np.int32)
        kr = keypoints[i, 0]
        kc = keypoints[i, 1]
        for j in range(256):
            pr0 = steered_pos0[j][0]
            pc0 = steered_pos0[j][1]
            pr1 = steered_pos1[j][0]
            pc1 = steered_pos1[j][1]

            if image[kr + pr0, kc + pc0] < image[kr + pr1, kc + pc1]:
                descriptors[i, j] = True
