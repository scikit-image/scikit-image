#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp


def _censure_dob_loop(double[:, ::1] image, cnp.int16_t n,
	                  double[:, ::1] integral_img,
	                  double[:, ::1] filtered_image,
	                  cnp.float_t inner_wt, cnp.float_t outer_wt):

    cdef Py_ssize_t i, j
    cdef double inner, outer

    for i in range(2 * n, image.shape[0] - 2 * n):
        for j in range(2 * n, image.shape[1] - 2 * n):
            inner = integral_img[i + n, j + n] + integral_img[i - n - 1, j - n - 1] - integral_img[i + n, j - n - 1] - integral_img[i - n - 1, j + n]
            outer = integral_img[i + 2 * n, j + 2 * n] + integral_img[i - 2 * n - 1, j - 2 * n - 1] - integral_img[i + 2 * n, j - 2 * n - 1] - integral_img[i - 2 * n - 1, j + 2 * n]
            filtered_image[i, j] = outer_wt * outer - (inner_wt + outer_wt) * inner
