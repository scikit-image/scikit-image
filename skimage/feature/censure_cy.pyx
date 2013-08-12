#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False


def _censure_dob_loop(Py_ssize_t n,
                      double[:, ::1] integral_img,
                      double[:, ::1] filtered_image,
                      double inner_weight, double outer_weight):
    # This function calculates the value in the DoB filtered image using
    # integral images. If r = right. l = left, u = up, d = down, the sum of
    # pixel values in the rectangle formed by (u, l), (u, r), (d, r), (d, l)
    # is calculated as I(d, r) + I(u - 1, l - 1) - I(u - 1, r) - I(d, l - 1).

    cdef Py_ssize_t i, j
    cdef double inner, outer
    cdef Py_ssize_t n2 = 2 * n
    cdef double total_weight = inner_weight + outer_weight

    for i in range(n2, integral_img.shape[0] - n2):
        for j in range(n2, integral_img.shape[1] - n2):
            inner = (integral_img[i + n, j + n]
                     + integral_img[i - n - 1, j - n - 1]
                     - integral_img[i + n, j - n - 1]
                     - integral_img[i - n - 1, j + n])

            outer = (integral_img[i + n2, j + n2]
                     + integral_img[i - n2 - 1, j - n2 - 1]
                     - integral_img[i + n2, j - n2 - 1]
                     - integral_img[i - n2 - 1, j + n2])

            filtered_image[i, j] = (outer_weight * outer
                                    - total_weight * inner)
