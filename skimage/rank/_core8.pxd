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

#---------------------------------------------------------------------------
# 8 bit core kernel
#---------------------------------------------------------------------------

cdef inline _core8(np.uint8_t kernel(int*, float, np.uint8_t),
np.ndarray[np.uint8_t, ndim=2] image,
np.ndarray[np.uint8_t, ndim=2] selem,
np.ndarray[np.uint8_t, ndim=2] mask,
np.ndarray[np.uint8_t, ndim=2] out,
char shift_x, char shift_y):
    """ Main loop, this function computes the histogram for each image point
    - data is uint8
    - result is uint8 casted
    """

    cdef int rows = image.shape[0]
    cdef int cols = image.shape[1]
    cdef int srows = selem.shape[0]
    cdef int scols = selem.shape[1]

    cdef int centre_r = int(selem.shape[0] / 2) + shift_y
    cdef int centre_c = int(selem.shape[1] / 2) + shift_x

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

    if out is None:
        out = np.zeros((rows, cols), dtype=np.uint8)
    else:
        out = np.ascontiguousarray(out)

    # create extended image and mask
    cdef int erows = rows+srows-1
    cdef int ecols = cols+scols-1

    cdef np.ndarray emask = np.zeros((erows, ecols), dtype=np.uint8)
    cdef np.ndarray eimage = np.zeros((erows, ecols), dtype=np.uint8)

    eimage[centre_r:rows+centre_r,centre_c:cols+centre_c] = image
    emask[centre_r:rows+centre_r,centre_c:cols+centre_c] = mask

    mask = np.ascontiguousarray(mask)

    # define pointers to the data
    cdef np.uint8_t* eimage_data = <np.uint8_t*>eimage.data
    cdef np.uint8_t* emask_data = <np.uint8_t*>emask.data

    cdef np.uint8_t* out_data = <np.uint8_t*>out.data
    cdef np.uint8_t* image_data = <np.uint8_t*>image.data
    cdef np.uint8_t* mask_data = <np.uint8_t*>mask.data

    # define local variable types
    cdef int r, c, rr, cc, s, value, local_max, i, even_row
    cdef float pop                                 # number of pixels actually inside the neighborhood (float)

    # allocate memory with malloc
    cdef int max_se = srows*scols

    # number of element in each attack border
    cdef int n_se_n, n_se_s, n_se_e, n_se_w

    # the current local histogram distribution
    cdef int* histo = <int*>malloc(256 * sizeof(int))

    # these lists contain the relative pixel row and column for each of the 4 attack borders
    # east, west, north and south
    # e.g. se_e_r lists the rows of the east structuring element border

    cdef int* se_e_r = <int*>malloc(max_se * sizeof(int))
    cdef int* se_e_c = <int*>malloc(max_se * sizeof(int))
    cdef int* se_w_r = <int*>malloc(max_se * sizeof(int))
    cdef int* se_w_c = <int*>malloc(max_se * sizeof(int))
    cdef int* se_n_r = <int*>malloc(max_se * sizeof(int))
    cdef int* se_n_c = <int*>malloc(max_se * sizeof(int))
    cdef int* se_s_r = <int*>malloc(max_se * sizeof(int))
    cdef int* se_s_c = <int*>malloc(max_se * sizeof(int))

    # build attack and release borders
    # by using difference along axis

    t = np.hstack((selem,np.zeros((selem.shape[0],1))))
    t_e = np.diff(t,axis=1)==-1

    t = np.hstack((np.zeros((selem.shape[0],1)),selem))
    t_w = np.diff(t,axis=1)==1

    t = np.vstack((selem,np.zeros((1,selem.shape[1]))))
    t_s = np.diff(t,axis=0)==-1

    t = np.vstack((np.zeros((1,selem.shape[1])),selem))
    t_n = np.diff(t,axis=0)==1

    n_se_n = n_se_s = n_se_e = n_se_w = 0

    for r in range(srows):
        for c in range(scols):
            if t_e[r,c]:
                se_e_r[n_se_e] = r - centre_r
                se_e_c[n_se_e] = c - centre_c
                n_se_e += 1
            if t_w[r,c]:
                se_w_r[n_se_w] = r - centre_r
                se_w_c[n_se_w] = c - centre_c
                n_se_w += 1
            if t_n[r,c]:
                se_n_r[n_se_n] = r - centre_r
                se_n_c[n_se_n] = c - centre_c
                n_se_n += 1
            if t_s[r,c]:
                se_s_r[n_se_s] = r - centre_r
                se_s_c[n_se_s] = c - centre_c
                n_se_s += 1

    # initial population and histogram
    for i in range(256):
        histo[i] = 0

    pop = 0

    for r in range(srows):
        for c in range(scols):
            rr = r
            cc = c
            if selem[r, c]:
                if emask_data[rr * ecols + cc]:
                    value = eimage_data[rr * ecols + cc]
                    histo[value] += 1
                    pop += 1.

    r = 0
    c = 0
    # kernel -------------------------------------------
    out_data[r * cols + c] = kernel(histo,pop,eimage_data[(r+centre_r) * ecols + c + centre_c])
    # kernel -------------------------------------------

    # main loop
    r = 0
    for even_row in range(0,rows,2):
        # ---> west to east
        for c in range(1,cols):
            for s in range(n_se_e):
                rr = r + se_e_r[s] + centre_r
                cc = c + se_e_c[s] + centre_c
                if emask_data[rr * ecols + cc]:
                    value = eimage_data[rr * ecols + cc]
                    histo[value] += 1
                    pop += 1.
            for s in range(n_se_w):
                rr = r + se_w_r[s] + centre_r
                cc = c + se_w_c[s] + centre_c - 1
                if emask_data[rr * ecols + cc]:
                    value = eimage_data[rr * ecols + cc]
                    histo[value] -= 1
                    pop -= 1.

            # kernel -------------------------------------------
            out_data[r * cols + c] = kernel(histo,pop,eimage_data[(r+centre_r) * ecols + c + centre_c])
            # kernel -------------------------------------------

        r += 1          # pass to the next row
        if r>=rows:
            break

    # ---> north to south
        for s in range(n_se_s):
            rr = r + se_s_r[s] + centre_r
            cc = c + se_s_c[s] + centre_c
            if emask_data[rr * ecols + cc]:
                value = eimage_data[rr * ecols + cc]
                histo[value] += 1
                pop += 1.
        for s in range(n_se_n):
            rr = r + se_n_r[s] + centre_r - 1
            cc = c + se_n_c[s] + centre_c
            if emask_data[rr * ecols + cc]:
                value = eimage_data[rr * ecols + cc]
                histo[value] -= 1
                pop -= 1.

        # kernel -------------------------------------------
        out_data[r * cols + c] = kernel(histo,pop,eimage_data[(r+centre_r) * ecols + c + centre_c])
        # kernel -------------------------------------------

        # ---> east to west
        for c in range(cols-2,-1,-1):
            for s in range(n_se_w):
                rr = r + se_w_r[s] + centre_r
                cc = c + se_w_c[s] + centre_c
                if emask_data[rr * ecols + cc]:
                    value = eimage_data[rr * ecols + cc]
                    histo[value] += 1
                    pop += 1.
            for s in range(n_se_e):
                rr = r + se_e_r[s] + centre_r
                cc = c + se_e_c[s] + centre_c + 1
                if emask_data[rr * ecols + cc]:
                    value = eimage_data[rr * ecols + cc]
                    histo[value] -= 1
                    pop -= 1.

            # kernel -------------------------------------------
            out_data[r * cols + c] = kernel(histo,pop,eimage_data[(r+centre_r) * ecols + c + centre_c])
            # kernel -------------------------------------------

        r += 1           # pass to the next row
        if r>=rows:
            break

        # ---> north to south
        for s in range(n_se_s):
            rr = r + se_s_r[s] + centre_r
            cc = c + se_s_c[s] + centre_c
            if emask_data[rr * ecols + cc]:
                value = eimage_data[rr * ecols + cc]
                histo[value] += 1
                pop += 1.
        for s in range(n_se_n):
            rr = r + se_n_r[s] + centre_r - 1
            cc = c + se_n_c[s] + centre_c
            if emask_data[rr * ecols + cc]:
                value = eimage_data[rr * ecols + cc]
                histo[value] -= 1
                pop -= 1.

        # kernel -------------------------------------------
        out_data[r * cols + c] = kernel(histo,pop,eimage_data[(r+centre_r) * ecols + c + centre_c])
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

    return out

