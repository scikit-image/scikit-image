#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np

from ..util import img_as_float


def _corner_fast(double[:, ::1] image, int n, double threshold):
    
    cdef int[:] rp = (np.round(3 * np.sin(2 * np.pi * np.arange(16, dtype=np.double) / 16))).astype(np.int32)
    cdef int[:] cp = (np.round(3 * np.cos(2 * np.pi * np.arange(16, dtype=np.double) / 16))).astype(np.int32)

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]

    cdef Py_ssize_t i, j, k, l, m

    cdef char[:] bins
    cdef int consecutive_count, speed_sum_b, speed_sum_d
    cdef int sp
    cdef double sum_b
    cdef double sum_d
    cdef double[:, ::1] corner_response = np.zeros((rows, cols), dtype=np.double)

    cdef double circle_intensity

    for i in range(3, rows - 3):
        for j in range(3, cols - 3):

            bins = np.zeros(16, dtype='S1')
            speed_sum_b = 0
            speed_sum_d = 0
            sum_b = 0
            sum_d = 0

            for k in range(16):
                circle_intensity = image[i + rp[k], j + cp[k]]
                if circle_intensity > image[i, j] + threshold:
                    # Brighter pixel
                    bins[k] = 'b'
                elif circle_intensity < image[i, j] - threshold:
                    # Darker pixel
                    bins[k] = 'd'
                else:
                    # Similar pixel
                    bins[k] = 's'

            # High speed test for n>=12
            if n >= 12:
                for k in range(4):
                    if bins[4 * k] == 'b':
                        speed_sum_b += 1
                    elif bins[4 * k] == 'd':
                        speed_sum_d += 1
                if speed_sum_d < 3 and speed_sum_b < 3:
                    continue

            consecutive_count = 0
            for l in range(15 + n):
                if bins[l % 16] == 'b':
                    consecutive_count += 1
                    if consecutive_count == n:
                        for m in range(16):
                            if bins[m] == 'b':
                                sum_b += image[i + rp[m], j + cp[m]] - image[i, j] - threshold
                            elif bins[m] == 'd':
                                sum_d += image[i, j] - image[i + rp[m], j + cp[m]] - threshold
                        # Finding the response of the corner
                        if sum_d > sum_b:
                            corner_response[i, j] = sum_d
                        else:
                            corner_response[i, j] = sum_b
                        break
                else:
                    consecutive_count = 0

            if corner_response[i, j] == 0:
                consecutive_count = 0
                for l in range(15 + n):
                    if bins[l % 16] == 'd':
                        consecutive_count += 1
                        if consecutive_count == n:
                            for m in range(16):
                                if bins[m] == 'b':
                                    sum_b += image[i + rp[m], j + cp[m]] - image[i, j] - threshold
                                elif bins[m] == 'd':
                                    sum_d += image[i, j] - image[i + rp[m], j + cp[m]] - threshold
                            # Finding the response of the corner
                            if sum_d > sum_b:
                                corner_response[i, j] = sum_d
                            else:
                                corner_response[i, j] = sum_b
                            break
                    else:
                        consecutive_count = 0

    return np.asarray(corner_response)
