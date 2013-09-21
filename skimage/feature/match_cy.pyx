import numpy as np


def _binary_cross_check_loop(Py_ssize_t[:] matched_keypoints1_index,
                             Py_ssize_t[:] matched_keypoints2_index,
                             double[:, ::1] distance, double threshold):
    cdef Py_ssize_t i
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t[:, ::1] matched_index = np.zeros((len(matched_keypoints1_index), 2), dtype=np.intp)

    for i in range(len(matched_keypoints1_index)):
        if (matched_keypoints2_index[matched_keypoints1_index[i]] == i and
            distance[i, matched_keypoints1_index[i]] < threshold):
            matched_index[count, 0] = i
            matched_index[count, 1] = matched_keypoints1_index[i]
            count += 1

    return np.asarray(matched_index[:count, :])
