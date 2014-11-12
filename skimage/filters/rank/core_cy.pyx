#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np

cimport numpy as cnp
from libc.stdlib cimport malloc, free


cdef inline dtype_t _max(dtype_t a, dtype_t b):
    return a if a >= b else b


cdef inline dtype_t _min(dtype_t a, dtype_t b):
    return a if a <= b else b


cdef inline void histogram_increment(Py_ssize_t* histo, double* pop,
                                     dtype_t value):
    histo[value] += 1
    pop[0] += 1


cdef inline void histogram_decrement(Py_ssize_t* histo, double* pop,
                                     dtype_t value):
    histo[value] -= 1
    pop[0] -= 1


cdef inline char is_in_mask(Py_ssize_t rows, Py_ssize_t cols,
                            Py_ssize_t r, Py_ssize_t c,
                            char* mask):
    """Check whether given coordinate is within image and mask is true."""
    if r < 0 or r > rows - 1 or c < 0 or c > cols - 1:
        return 0
    else:
        if mask[r * cols + c]:
            return 1
        else:
            return 0


cdef void _core(void kernel(dtype_t_out*, Py_ssize_t, Py_ssize_t*, double,
                            dtype_t, Py_ssize_t, Py_ssize_t, double,
                            double, Py_ssize_t, Py_ssize_t),
                dtype_t[:, ::1] image,
                char[:, ::1] selem,
                char[:, ::1] mask,
                dtype_t_out[:, :, ::1] out,
                signed char shift_x, signed char shift_y,
                double p0, double p1,
                Py_ssize_t s0, Py_ssize_t s1,
                Py_ssize_t max_bin) except *:
    """Compute histogram for each pixel neighborhood, apply kernel function and
    use kernel function return value for output image.
    """

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]
    cdef Py_ssize_t srows = selem.shape[0]
    cdef Py_ssize_t scols = selem.shape[1]
    cdef Py_ssize_t odepth = out.shape[2]

    cdef Py_ssize_t centre_r = <Py_ssize_t>(selem.shape[0] / 2) + shift_y
    cdef Py_ssize_t centre_c = <Py_ssize_t>(selem.shape[1] / 2) + shift_x

    # check that structuring element center is inside the element bounding box
    assert centre_r >= 0
    assert centre_c >= 0
    assert centre_r < srows
    assert centre_c < scols

    # add 1 to ensure maximum value is included in histogram -> range(max_bin)
    max_bin += 1

    cdef Py_ssize_t mid_bin = max_bin / 2

    # define pointers to the data
    cdef char* mask_data = &mask[0, 0]

    # define local variable types
    cdef Py_ssize_t r, c, rr, cc, s, value, local_max, i, even_row

    # number of pixels actually inside the neighborhood (double)
    cdef double pop = 0

    # the current local histogram distribution
    cdef Py_ssize_t* histo = <Py_ssize_t*>malloc(max_bin * sizeof(Py_ssize_t))
    for i in range(max_bin):
        histo[i] = 0

    # these lists contain the relative pixel row and column for each of the 4
    # attack borders east, west, north and south e.g. se_e_r lists the rows of
    # the east structuring element border

    cdef Py_ssize_t max_se = srows * scols

    # number of element in each attack border
    cdef Py_ssize_t num_se_n, num_se_s, num_se_e, num_se_w
    num_se_n = num_se_s = num_se_e = num_se_w = 0

    cdef Py_ssize_t* se_e_r = <Py_ssize_t*>malloc(max_se * sizeof(Py_ssize_t))
    cdef Py_ssize_t* se_e_c = <Py_ssize_t*>malloc(max_se * sizeof(Py_ssize_t))
    cdef Py_ssize_t* se_w_r = <Py_ssize_t*>malloc(max_se * sizeof(Py_ssize_t))
    cdef Py_ssize_t* se_w_c = <Py_ssize_t*>malloc(max_se * sizeof(Py_ssize_t))
    cdef Py_ssize_t* se_n_r = <Py_ssize_t*>malloc(max_se * sizeof(Py_ssize_t))
    cdef Py_ssize_t* se_n_c = <Py_ssize_t*>malloc(max_se * sizeof(Py_ssize_t))
    cdef Py_ssize_t* se_s_r = <Py_ssize_t*>malloc(max_se * sizeof(Py_ssize_t))
    cdef Py_ssize_t* se_s_c = <Py_ssize_t*>malloc(max_se * sizeof(Py_ssize_t))

    # build attack and release borders by using difference along axis
    t = np.hstack((selem, np.zeros((selem.shape[0], 1))))
    cdef unsigned char[:, :] t_e = (np.diff(t, axis=1) < 0).view(np.uint8)

    t = np.hstack((np.zeros((selem.shape[0], 1)), selem))
    cdef unsigned char[:, :] t_w = (np.diff(t, axis=1) > 0).view(np.uint8)

    t = np.vstack((selem, np.zeros((1, selem.shape[1]))))
    cdef unsigned char[:, :] t_s = (np.diff(t, axis=0) < 0).view(np.uint8)

    t = np.vstack((np.zeros((1, selem.shape[1])), selem))
    cdef unsigned char[:, :] t_n = (np.diff(t, axis=0) > 0).view(np.uint8)

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

    for r in range(srows):
        for c in range(scols):
            rr = r - centre_r
            cc = c - centre_c
            if selem[r, c]:
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_increment(histo, &pop, image[rr, cc])

    r = 0
    c = 0
    kernel(&out[r, c, 0], odepth, histo, pop, image[r, c], max_bin, mid_bin,
           p0, p1, s0, s1)

    # main loop
    r = 0
    for even_row in range(0, rows, 2):

        # ---> west to east
        for c in range(1, cols):
            for s in range(num_se_e):
                rr = r + se_e_r[s]
                cc = c + se_e_c[s]
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_increment(histo, &pop, image[rr, cc])

            for s in range(num_se_w):
                rr = r + se_w_r[s]
                cc = c + se_w_c[s] - 1
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_decrement(histo, &pop, image[rr, cc])

            kernel(&out[r, c, 0], odepth, histo, pop, image[r, c], max_bin,
                   mid_bin, p0, p1, s0, s1)

        r += 1  # pass to the next row
        if r >= rows:
            break

        # ---> north to south
        for s in range(num_se_s):
            rr = r + se_s_r[s]
            cc = c + se_s_c[s]
            if is_in_mask(rows, cols, rr, cc, mask_data):
                histogram_increment(histo, &pop, image[rr, cc])

        for s in range(num_se_n):
            rr = r + se_n_r[s] - 1
            cc = c + se_n_c[s]
            if is_in_mask(rows, cols, rr, cc, mask_data):
                histogram_decrement(histo, &pop, image[rr, cc])

        kernel(&out[r, c, 0], odepth, histo, pop, image[r, c], max_bin,
               mid_bin, p0, p1, s0, s1)

        # ---> east to west
        for c in range(cols - 2, -1, -1):
            for s in range(num_se_w):
                rr = r + se_w_r[s]
                cc = c + se_w_c[s]
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_increment(histo, &pop, image[rr, cc])

            for s in range(num_se_e):
                rr = r + se_e_r[s]
                cc = c + se_e_c[s] + 1
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_decrement(histo, &pop, image[rr, cc])

            kernel(&out[r, c, 0], odepth, histo, pop, image[r, c], max_bin,
                   mid_bin, p0, p1, s0, s1)

        r += 1  # pass to the next row
        if r >= rows:
            break

        # ---> north to south
        for s in range(num_se_s):
            rr = r + se_s_r[s]
            cc = c + se_s_c[s]
            if is_in_mask(rows, cols, rr, cc, mask_data):
                histogram_increment(histo, &pop, image[rr, cc])

        for s in range(num_se_n):
            rr = r + se_n_r[s] - 1
            cc = c + se_n_c[s]
            if is_in_mask(rows, cols, rr, cc, mask_data):
                histogram_decrement(histo, &pop, image[rr, cc])

        kernel(&out[r, c, 0], odepth, histo, pop, image[r, c],
               max_bin, mid_bin, p0, p1, s0, s1)

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
