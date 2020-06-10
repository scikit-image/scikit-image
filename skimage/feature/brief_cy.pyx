#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
cnp.import_array()

from .._shared.fused_numerics cimport np_floats


def _brief_loop(np_floats[:, ::1] image, unsigned char[:, ::1] descriptors,
                Py_ssize_t[:, ::1] keypoints,
                int[:, ::1] pos0, int[:, ::1] pos1):

    cdef Py_ssize_t k, d, kr, kc, pr0, pr1, pc0, pc1

    with nogil:
        for p in range(pos0.shape[0]):
            pr0 = pos0[p, 0]
            pc0 = pos0[p, 1]
            pr1 = pos1[p, 0]
            pc1 = pos1[p, 1]
            for k in range(keypoints.shape[0]):
                kr = keypoints[k, 0]
                kc = keypoints[k, 1]
                if image[kr + pr0, kc + pc0] < image[kr + pr1, kc + pc1]:
                    descriptors[k, p] = True
