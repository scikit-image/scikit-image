#cython: cdivision=True
#cython: boundscheck=True
#cython: nonecheck=True
#cython: wraparound=True

cimport numpy as cnp
import numpy as np


def _censure_dob_loop(double[:, ::1] image, cnp.int16_t n,
	                  double[:, ::1] integral_img,
	                  double[:, ::1] filtered_image,
	                  double inner_wt, double outer_wt):

    cdef Py_ssize_t i, j
    cdef double inner, outer

    for i in range(2 * n, image.shape[0] - 2 * n):
        for j in range(2 * n, image.shape[1] - 2 * n):
            inner = integral_img[i + n, j + n] + integral_img[i - n - 1, j - n - 1] - integral_img[i + n, j - n - 1] - integral_img[i - n - 1, j + n]
            outer = integral_img[i + 2 * n, j + 2 * n] + integral_img[i - 2 * n - 1, j - 2 * n - 1] - integral_img[i + 2 * n, j - 2 * n - 1] - integral_img[i - 2 * n - 1, j + 2 * n]
            filtered_image[i, j] = outer_wt * outer - (inner_wt + outer_wt) * inner


def _slanted_integral_image(double[:, ::1] image,
                            double[:, ::1] integral_img):

    cdef Py_ssize_t i, j
    cdef double[:] left_sum = np.zeros(image.shape[0], dtype=np.float)

    flipped_lr = np.asarray(image[:, ::-1])
    for i in range(image.shape[1] - image.shape[0], image.shape[1]):
        left_sum[image.shape[1] - 1 - i] = np.sum(flipped_lr.diagonal(i))
    left_sum_np = np.asarray(left_sum)

    left_sum_np = left_sum_np.cumsum(0)

    right_sum_np = np.sum(image, 1).cumsum(0)

    print '1'
    for i in range(image.shape[0]):
        image[i, 0] = left_sum_np[i]
        image[i, -1] = right_sum_np[i]

    print '2'
    for i in range(1, integral_img.shape[0]):
        for j in range(integral_img.shape[1]):
            integral_img[i, j] = image[i - 1, j]

    print '3'
    for i in range(1, integral_img.shape[0]):
        for j in range(1, integral_img.shape[1] - 1):
            integral_img[i, j] += integral_img[i, j - 1] + integral_img[i - 1, j + 1] - integral_img[i - 1, j]
    print '4'



def _censure_octagon_loop(double[:, ::1] image, double[:, ::1] integral_img,
                          double[:, ::1] integral_img1,
                          double[:, ::1] integral_img2,
                          double[:, ::1] integral_img3,
                          double[:, ::1] integral_img4,
                          double[:, ::1] filtered_image,
                          double outer_wt, double inner_wt,
                          int mo, int no, int mi, int ni):
                    
    cdef Py_ssize_t i, j, o_m, i_m, o_set, i_set

    o_m = (mo - 1) / 2
    i_m = (mi - 1) / 2
    o_set = o_m + no
    i_set = i_m + ni
    print '5'
    for i in range(o_set + 1, image.shape[0] - o_set - 1):
        for j in range(o_set + 1, image.shape[1] - o_set - 1):
            outer = integral_img1[i + o_set, j + o_m] - integral_img1[i + o_m, j + o_set] - integral_img[i + o_set, j - o_m] + integral_img[i + o_m, j - o_m]
            outer += integral_img[i + o_m - 1, j + o_m - 1] - integral_img[i - o_m, j + o_m - 1] - integral_img[i + o_m - 1, j - o_m] + integral_img[i - o_m, j - o_m]
            outer += integral_img4[i + o_m, j - o_set] - integral_img4[i + o_set, j - o_m] - integral_img[i - o_m, j - o_m + 1] + integral_img[i - o_m, j - o_set - 1]
            outer += integral_img2[i - o_set, j - o_m] - integral_img2[i - o_m, j - o_set] - integral_img[i - o_m + 1, -1] - integral_img[i - o_set - 1, j + o_m - 1] + integral_img[i - o_m + 1, j + o_m - 1] + integral_img[i - o_set - 1, -1]
            outer += integral_img3[i - o_m, j + o_set] - integral_img3[i - o_set, j + o_m] - integral_img[-1, j + o_set + 1] - integral_img[i + o_m - 1, j + o_m] + integral_img[-1, j + o_m] + integral_img[i + o_m - 1, j + o_set + 1]

            inner = integral_img1[i + i_set, j + i_m] - integral_img1[i + i_m, j + i_set] - integral_img[i + i_set, j - i_m] + integral_img[i + i_m, j - i_m]
            inner += integral_img[i + i_m - 1, j + i_m - 1] - integral_img[i - i_m, j + i_m - 1] - integral_img[i + i_m - 1, j - i_m] + integral_img[i - i_m, j - i_m]
            inner += integral_img4[i + i_m, j - i_set] - integral_img4[i + i_set, j - i_m] - integral_img[i - i_m, j - i_m + 1] + integral_img[i - i_m, j - i_set - 1]
            inner += integral_img2[i - i_set, j - i_m] - integral_img2[i - i_m, j - i_set] - integral_img[i - i_m + 1, -1] - integral_img[i - i_set - 1, j + i_m - 1] + integral_img[i - i_m + 1, j + i_m - 1] + integral_img[i - i_set - 1, -1]
            inner += integral_img3[i - i_m, j + i_set] - integral_img3[i - i_set, j + i_m] - integral_img[-1, j + i_set + 1] - integral_img[i + i_m - 1, j + i_m] + integral_img[-1, j + i_m] + integral_img[i + i_m - 1, j + i_set + 1]

            filtered_image[i, j] = outer_wt * outer - (outer_wt + inner_wt) * inner
    print '6'
