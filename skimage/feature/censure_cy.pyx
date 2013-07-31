#cython: cdivision=True
#cython: boundscheck=True
#cython: nonecheck=True
#cython: wraparound=True

cimport numpy as cnp
import numpy as np


def _censure_dob_loop(double[:, ::1] image, Py_ssize_t n,
                      double[:, ::1] integral_img,
                      double[:, ::1] filtered_image,
                      double inner_weight, double outer_weight):

    cdef Py_ssize_t i, j
    cdef double inner, outer

    for i in range(2 * n, image.shape[0] - 2 * n):
        for j in range(2 * n, image.shape[1] - 2 * n):
            inner = integral_img[i + n, j + n] + integral_img[i - n - 1, j - n - 1] - integral_img[i + n, j - n - 1] - integral_img[i - n - 1, j + n]
            outer = integral_img[i + 2 * n, j + 2 * n] + integral_img[i - 2 * n - 1, j - 2 * n - 1] - integral_img[i + 2 * n, j - 2 * n - 1] - integral_img[i - 2 * n - 1, j + 2 * n]
            filtered_image[i, j] = outer_weight * outer - (inner_weight + outer_weight) * inner
