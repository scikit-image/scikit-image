#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free


def dilate(np.ndarray[np.uint8_t, ndim=2] image,
           np.ndarray[np.uint8_t, ndim=2] selem,
           np.ndarray[np.uint8_t, ndim=2] out=None,
           char shift_x=0, char shift_y=0):

    cdef ssize_t rows = image.shape[0]
    cdef ssize_t cols = image.shape[1]
    cdef ssize_t srows = selem.shape[0]
    cdef ssize_t scols = selem.shape[1]

    cdef ssize_t centre_r = int(selem.shape[0] / 2) - shift_y
    cdef ssize_t centre_c = int(selem.shape[1] / 2) - shift_x

    image = np.ascontiguousarray(image)
    if out is None:
        out = np.zeros((rows, cols), dtype=np.uint8)
    else:
        out = np.ascontiguousarray(out)

    cdef np.uint8_t* out_data = <np.uint8_t*>out.data
    cdef np.uint8_t* image_data = <np.uint8_t*>image.data

    cdef ssize_t r, c, rr, cc, s, value, local_max

    cdef ssize_t selem_num = np.sum(selem != 0)
    cdef ssize_t* sr = <ssize_t*>malloc(selem_num * sizeof(ssize_t))
    cdef ssize_t* sc = <ssize_t*>malloc(selem_num * sizeof(ssize_t))

    s = 0
    for r in range(srows):
        for c in range(scols):
            if selem[r, c] != 0:
                sr[s] = r - centre_r
                sc[s] = c - centre_c
                s += 1

    for r in range(rows):
        for c in range(cols):
            local_max = 0
            for s in range(selem_num):
                rr = r + sr[s]
                cc = c + sc[s]
                if 0 <= rr < rows and 0 <= cc < cols:
                    value = image_data[rr * cols + cc]
                    if value > local_max:
                        local_max = value

            out_data[r * cols + c] = local_max

    free(sr)
    free(sc)

    return out


def erode(np.ndarray[np.uint8_t, ndim=2] image,
          np.ndarray[np.uint8_t, ndim=2] selem,
          np.ndarray[np.uint8_t, ndim=2] out=None,
          char shift_x=0, char shift_y=0):

    cdef ssize_t rows = image.shape[0]
    cdef ssize_t cols = image.shape[1]
    cdef ssize_t srows = selem.shape[0]
    cdef ssize_t scols = selem.shape[1]

    cdef ssize_t centre_r = int(selem.shape[0] / 2) - shift_y
    cdef ssize_t centre_c = int(selem.shape[1] / 2) - shift_x

    image = np.ascontiguousarray(image)
    if out is None:
        out = np.zeros((rows, cols), dtype=np.uint8)
    else:
        out = np.ascontiguousarray(out)

    cdef np.uint8_t* out_data = <np.uint8_t*>out.data
    cdef np.uint8_t* image_data = <np.uint8_t*>image.data

    cdef int r, c, rr, cc, s, value, local_min

    cdef ssize_t selem_num = np.sum(selem != 0)
    cdef ssize_t* sr = <ssize_t*>malloc(selem_num * sizeof(ssize_t))
    cdef ssize_t* sc = <ssize_t*>malloc(selem_num * sizeof(ssize_t))

    s = 0
    for r in range(srows):
        for c in range(scols):
            if selem[r, c] != 0:
                sr[s] = r - centre_r
                sc[s] = c - centre_c
                s += 1

    for r in range(rows):
        for c in range(cols):
            local_min = 255
            for s in range(selem_num):
                rr = r + sr[s]
                cc = c + sc[s]
                if 0 <= rr < rows and 0 <= cc < cols:
                    value = image_data[rr * cols + cc]
                    if value < local_min:
                        local_min = value

            out_data[r * cols + c] = local_min

    free(sr)
    free(sc)

    return out
