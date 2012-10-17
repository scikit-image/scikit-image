""" to compile this use:
>>> python setup.py build_ext --inplace

to generate html report use:
>>> cython -a core8.pxd
"""

#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

# generic cdef functions
cdef inline np.uint8_t uint8_max(np.uint8_t a, np.uint8_t b):
    return a if a >= b else b
cdef inline np.uint8_t uint8_min(np.uint8_t a, np.uint8_t b):
    return a if a <= b else b


#---------------------------------------------------------------------------
# 8 bit core kernel
#---------------------------------------------------------------------------

cdef inline void histogram_increment(Py_ssize_t * histo, float * pop, np.uint8_t value):
    histo[value] += 1
    pop[0] += 1.

cdef inline void histogram_decrement(Py_ssize_t * histo, float * pop, np.uint8_t value):
    histo[value] -= 1
    pop[0] -= 1.

cdef inline np.uint8_t is_in_mask(Py_ssize_t rows, Py_ssize_t cols, Py_ssize_t r, Py_ssize_t c, np.uint8_t * mask):
    """ returns 1 if given(r,c) coordinate are within the image frame ([0-rows],[0-cols]) and
        inside the given mask
        returns 0 otherwise
    """
    if r < 0 or r > rows - 1 or c < 0 or c > cols - 1:
        return 0
    else:
        if mask[r * cols + c]:
            return 1
        else:
            return 0

cdef inline _core8(
    np.uint8_t kernel(Py_ssize_t * , float, np.uint8_t, float, float, Py_ssize_t, Py_ssize_t),
    np.ndarray[np.uint8_t, ndim=2] image,
    np.ndarray[np.uint8_t, ndim=2] selem,
    np.ndarray[np.uint8_t, ndim=2] mask,
    np.ndarray[np.uint8_t, ndim=2] out,
        char shift_x, char shift_y, float p0, float p1, Py_ssize_t s0, Py_ssize_t s1):
    """ Main loop, this function computes the histogram for each image point
    - data is uint8
    - result is uint8 casted
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

    image = np.ascontiguousarray(image)

    if mask is None:
        mask = np.ones((rows, cols), dtype=np.uint8)
    else:
        mask = np.ascontiguousarray(mask)

    if image is out:
        raise NotImplementedError("Cannot perform rank operation in place.")

    if out is None:
        out = np.zeros((rows, cols), dtype=np.uint8)
    else:
        out = np.ascontiguousarray(out)

    mask = np.ascontiguousarray(mask)

    # define pointers to the data

    cdef np.uint8_t * out_data = <np.uint8_t * >out.data
    cdef np.uint8_t * image_data = <np.uint8_t * >image.data
    cdef np.uint8_t * mask_data = <np.uint8_t * >mask.data

    # define local variable types
    cdef Py_ssize_t r, c, rr, cc, s, value, local_max, i, even_row

    # number of pixels actually inside the neighborhood (float)
    cdef float pop

    # allocate memory with malloc
    cdef Py_ssize_t max_se = srows * scols

    # number of element in each attack border
    cdef Py_ssize_t num_se_n, num_se_s, num_se_e, num_se_w

    # the current local histogram distribution
    cdef Py_ssize_t * histo = <Py_ssize_t * >malloc(256 * sizeof(Py_ssize_t))

    # these lists contain the relative pixel row and column for each of the 4 attack borders
    # east, west, north and south
    # e.g. se_e_r lists the rows of the east structuring element border

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
    t_e = np.diff(t, axis=1) == -1

    t = np.hstack((np.zeros((selem.shape[0], 1)), selem))
    t_w = np.diff(t, axis=1) == 1

    t = np.vstack((selem, np.zeros((1, selem.shape[1]))))
    t_s = np.diff(t, axis=0) == -1

    t = np.vstack((np.zeros((1, selem.shape[1])), selem))
    t_n = np.diff(t, axis=0) == 1

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

    # initial population and histogram (kernel is centered on the first row and column)
    for i in range(256):
        histo[i] = 0

    pop = 0

    for r in range(srows):
        for c in range(scols):
            rr = r - centre_r
            cc = c - centre_c
            if selem[r, c]:
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_increment(histo, & pop, image_data[rr * cols + cc])

    r = 0
    c = 0
    # kernel --------------------------------------------------------------------
    out_data[r * cols + c] = kernel(histo, pop, image_data[r * cols +
                                    c], p0, p1, s0, s1)
    # kernel --------------------------------------------------------------------

    # main loop
    r = 0
    for even_row in range(0, rows, 2):
        # ---> west to east
        for c in range(1, cols):
            for s in range(num_se_e):
                rr = r + se_e_r[s]
                cc = c + se_e_c[s]
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_increment(histo, & pop, image_data[rr * cols + cc])

            for s in range(num_se_w):
                rr = r + se_w_r[s]
                cc = c + se_w_c[s] - 1
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_decrement(histo, & pop, image_data[rr * cols + cc])

            # kernel --------------------------------------------------------------------
            out_data[r * cols + c] = kernel(
                histo, pop, image_data[r * cols + c], p0, p1, s0, s1)
            # kernel --------------------------------------------------------------------

        r += 1          # pass to the next row
        if r >= rows:
            break

            # ---> north to south
        for s in range(num_se_s):
            rr = r + se_s_r[s]
            cc = c + se_s_c[s]
            if is_in_mask(rows, cols, rr, cc, mask_data):
                histogram_increment(histo, & pop, image_data[rr * cols + cc])

        for s in range(num_se_n):
            rr = r + se_n_r[s] - 1
            cc = c + se_n_c[s]
            if is_in_mask(rows, cols, rr, cc, mask_data):
                histogram_decrement(histo, & pop, image_data[rr * cols + cc])

        # kernel --------------------------------------------------------------------
        out_data[r * cols + c] = kernel(histo, pop, image_data[r *
                                        cols + c], p0, p1, s0, s1)
        # kernel --------------------------------------------------------------------

        # ---> east to west
        for c in range(cols - 2, -1, -1):
            for s in range(num_se_w):
                rr = r + se_w_r[s]
                cc = c + se_w_c[s]
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_increment(histo, & pop, image_data[rr * cols + cc])

            for s in range(num_se_e):
                rr = r + se_e_r[s]
                cc = c + se_e_c[s] + 1
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_decrement(histo, & pop, image_data[rr * cols + cc])

            # kernel --------------------------------------------------------------------
            out_data[r * cols + c] = kernel(
                histo, pop, image_data[r * cols + c], p0, p1, s0, s1)
            # kernel --------------------------------------------------------------------

        r += 1           # pass to the next row
        if r >= rows:
            break

        # ---> north to south
        for s in range(num_se_s):
            rr = r + se_s_r[s]
            cc = c + se_s_c[s]
            if is_in_mask(rows, cols, rr, cc, mask_data):
                histogram_increment(histo, & pop, image_data[rr * cols + cc])

        for s in range(num_se_n):
            rr = r + se_n_r[s] - 1
            cc = c + se_n_c[s]
            if is_in_mask(rows, cols, rr, cc, mask_data):
                histogram_decrement(histo, & pop, image_data[rr * cols + cc])

        # kernel --------------------------------------------------------------------
        out_data[r * cols + c] = kernel(histo, pop, image_data[r *
                                        cols + c], p0, p1, s0, s1)
        # kernel --------------------------------------------------------------------

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

    return out
