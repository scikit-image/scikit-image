#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np

cimport numpy as cnp
from libc.stdlib cimport malloc, free
from _core8 cimport is_in_mask


cdef inline int int_max(int a, int b):
    return a if a >= b else b


cdef inline int int_min(int a, int b):
    return a if a <= b else b


cdef inline void histogram_increment(Py_ssize_t * histo, float * pop,
                                     dtype_t value):
    histo[value] += 1
    pop[0] += 1


cdef inline void histogram_decrement(Py_ssize_t * histo, float * pop,
                                     dtype_t value):
    histo[value] -= 1
    pop[0] -= 1


cdef void _core16(dtype_t kernel(Py_ssize_t *, float, dtype_t,
                                 Py_ssize_t, Py_ssize_t, Py_ssize_t, float,
                                 float, Py_ssize_t, Py_ssize_t),
                  cnp.ndarray[dtype_t, ndim=2] image,
                  cnp.ndarray[cnp.uint8_t, ndim=2] selem,
                  cnp.ndarray[cnp.uint8_t, ndim=2] mask,
                  cnp.ndarray[dtype_t, ndim=2] out,
                  char shift_x, char shift_y, Py_ssize_t bitdepth,
                  float p0, float p1, Py_ssize_t s0, Py_ssize_t s1) except *:
    """Compute histogram for each pixel neighborhood, apply kernel function and
    use kernel function return value for output image.
    """

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]
    cdef Py_ssize_t srows = selem.shape[0]
    cdef Py_ssize_t scols = selem.shape[1]

    cdef Py_ssize_t centre_r = int(selem.shape[0] / 2) + shift_y
    cdef Py_ssize_t centre_c = int(selem.shape[1] / 2) + shift_x

    # check that structuring element center is inside the element bounding box
    assert centre_r >= 0
    assert centre_c >= 0
    assert centre_r < srows
    assert centre_c < scols
    assert bitdepth in range(2, 13)

    maxbin_list = [0, 0, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    midbin_list = [0, 0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    # set maxbin and midbin
    cdef Py_ssize_t maxbin = maxbin_list[bitdepth]
    cdef Py_ssize_t midbin = midbin_list[bitdepth]

    assert (image < maxbin).all()

    # define pointers to the data
    cdef dtype_t * out_data = <dtype_t * >out.data
    cdef dtype_t * image_data = <dtype_t * >image.data
    cdef cnp.uint8_t * mask_data = <cnp.uint8_t * >mask.data

    # define local variable types
    cdef Py_ssize_t r, c, rr, cc, s, value, local_max, i, even_row
    # number of pixels actually inside the neighborhood (float)
    cdef float pop

    # allocate memory with malloc
    cdef Py_ssize_t max_se = srows * scols

    # number of element in each attack border
    cdef Py_ssize_t num_se_n, num_se_s, num_se_e, num_se_w

    # the current local histogram distribution
    cdef Py_ssize_t * histo = <Py_ssize_t * >malloc(maxbin * sizeof(Py_ssize_t))

    # these lists contain the relative pixel row and column for each of the 4
    # attack borders east, west, north and south e.g. se_e_r lists the rows of
    # the east structuring element border
    cdef Py_ssize_t * se_e_r = <Py_ssize_t * >malloc(max_se * sizeof(Py_ssize_t))
    cdef Py_ssize_t * se_e_c = <Py_ssize_t * >malloc(max_se * sizeof(Py_ssize_t))
    cdef Py_ssize_t * se_w_r = <Py_ssize_t * >malloc(max_se * sizeof(Py_ssize_t))
    cdef Py_ssize_t * se_w_c = <Py_ssize_t * >malloc(max_se * sizeof(Py_ssize_t))
    cdef Py_ssize_t * se_n_r = <Py_ssize_t * >malloc(max_se * sizeof(Py_ssize_t))
    cdef Py_ssize_t * se_n_c = <Py_ssize_t * >malloc(max_se * sizeof(Py_ssize_t))
    cdef Py_ssize_t * se_s_r = <Py_ssize_t * >malloc(max_se * sizeof(Py_ssize_t))
    cdef Py_ssize_t * se_s_c = <Py_ssize_t * >malloc(max_se * sizeof(Py_ssize_t))

    # build attack and release borders
    # by using difference along axis
    t = np.hstack((selem, np.zeros((selem.shape[0], 1))))
    t_e = np.diff(t, axis=1) < 0

    t = np.hstack((np.zeros((selem.shape[0], 1)), selem))
    t_w = np.diff(t, axis=1) > 0

    t = np.vstack((selem, np.zeros((1, selem.shape[1]))))
    t_s = np.diff(t, axis=0) < 0

    t = np.vstack((np.zeros((1, selem.shape[1])), selem))
    t_n = np.diff(t, axis=0) > 0

    num_se_n = num_se_s = num_se_e = num_se_w = 0

    for r in range(srows):
        for c in range(scols):
            if t_e[r, c]:
                se_e_r[num_se_e] = r - centre_r
                se_e_c[num_se_e] = c - centre_c
                num_se_e += 1
            if t_w[r, c]:
                se_w_r[num_se_w] = r - centre_r
                se_w_c[num_se_w] = c - centre_c
                num_se_w += 1
            if t_n[r, c]:
                se_n_r[num_se_n] = r - centre_r
                se_n_c[num_se_n] = c - centre_c
                num_se_n += 1
            if t_s[r, c]:
                se_s_r[num_se_s] = r - centre_r
                se_s_c[num_se_s] = c - centre_c
                num_se_s += 1

    # initial population and histogram
    for i in range(maxbin):
        histo[i] = 0

    pop = 0

    for r in range(srows):
        for c in range(scols):
            rr = r - centre_r
            cc = c - centre_c
            if selem[r, c]:
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_increment(histo, &pop, image_data[rr * cols + cc])

    r = 0
    c = 0
    # kernel -------------------------------------------
    out_data[r * cols + c] = kernel(histo, pop, image_data[r * cols + c],
        bitdepth, maxbin, midbin, p0, p1, s0, s1)
    # kernel -------------------------------------------

    # main loop
    r = 0
    for even_row in range(0, rows, 2):
        # ---> west to east
        for c in range(1, cols):
            for s in range(num_se_e):
                rr = r + se_e_r[s]
                cc = c + se_e_c[s]
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_increment(histo, &pop, image_data[rr * cols + cc])

            for s in range(num_se_w):
                rr = r + se_w_r[s]
                cc = c + se_w_c[s] - 1
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_decrement(histo, &pop, image_data[rr * cols + cc])

            # kernel -------------------------------------------
            out_data[r * cols + c] = kernel(
                histo, pop, image_data[r * cols + c],
                bitdepth, maxbin, midbin, p0, p1, s0, s1)
            # kernel -------------------------------------------

        r += 1          # pass to the next row
        if r >= rows:
            break

            # ---> north to south
        for s in range(num_se_s):
            rr = r + se_s_r[s]
            cc = c + se_s_c[s]
            if is_in_mask(rows, cols, rr, cc, mask_data):
                histogram_increment(histo, &pop, image_data[rr * cols + cc])

        for s in range(num_se_n):
            rr = r + se_n_r[s] - 1
            cc = c + se_n_c[s]
            if is_in_mask(rows, cols, rr, cc, mask_data):
                histogram_decrement(histo, &pop, image_data[rr * cols + cc])

        # kernel -------------------------------------------
        out_data[r * cols + c] = kernel(histo, pop, image_data[r * cols + c],
            bitdepth, maxbin, midbin, p0, p1, s0, s1)
        # kernel -------------------------------------------

        # ---> east to west
        for c in range(cols - 2, -1, -1):
            for s in range(num_se_w):
                rr = r + se_w_r[s]
                cc = c + se_w_c[s]
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_increment(histo, &pop, image_data[rr * cols + cc])

            for s in range(num_se_e):
                rr = r + se_e_r[s]
                cc = c + se_e_c[s] + 1
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_decrement(histo, &pop, image_data[rr * cols + cc])

            # kernel -------------------------------------------
            out_data[r * cols + c] = kernel(
                histo, pop, image_data[r * cols + c],
                bitdepth, maxbin, midbin, p0, p1, s0, s1)
            # kernel -------------------------------------------

        r += 1           # pass to the next row
        if r >= rows:
            break

        # ---> north to south
        for s in range(num_se_s):
            rr = r + se_s_r[s]
            cc = c + se_s_c[s]
            if is_in_mask(rows, cols, rr, cc, mask_data):
                histogram_increment(histo, &pop, image_data[rr * cols + cc])

        for s in range(num_se_n):
            rr = r + se_n_r[s] - 1
            cc = c + se_n_c[s]
            if is_in_mask(rows, cols, rr, cc, mask_data):
                histogram_decrement(histo, &pop, image_data[rr * cols + cc])

        # kernel -------------------------------------------
        out_data[r * cols + c] = kernel(histo, pop, image_data[r * cols + c],
            bitdepth, maxbin, midbin, p0, p1, s0, s1)
        # kernel -------------------------------------------

    # release memory allocated by malloc

    free(se_e_r)
    free(se_e_c)
    free(se_w_r)
    free(se_w_c)
    free(se_n_r)
    free(se_n_c)
    free(se_s_r)
    free(se_s_c)

    free(histo)
