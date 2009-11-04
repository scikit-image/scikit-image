# ColorMixer function implementations
import numpy as np
cimport numpy as np

import cython

@cython.boundscheck(False)
def add(np.ndarray[np.uint8_t, ndim=3] img,
        np.ndarray[np.uint8_t, ndim=3] stateimg,
        int channel, int ammount):

    cdef int height = img.shape[0]
    cdef int width = img.shape[1]
    cdef int k = channel
    cdef int n = ammount

    cdef np.int16_t op_result

    cdef int i, j
    for i in range(height):
        for j in range(width):
            op_result = <np.int16_t>(stateimg[i,j,k] + n)
            if op_result > 255:
                img[i, j, k] = 255
            elif op_result < 0:
                img[i, j, k] = 0
            else:
                img[i, j, k] = <np.uint8_t>op_result

@cython.boundscheck(False)
def multiply(np.ndarray[np.uint8_t, ndim=3] img,
             np.ndarray[np.uint8_t, ndim=3] stateimg,
             int channel, float ammount):

    cdef int height = img.shape[0]
    cdef int width = img.shape[1]
    cdef int k = channel
    cdef float n = ammount

    cdef float op_result

    cdef int i, j
    for i in range(height):
        for j in range(width):
            op_result = <float>(stateimg[i,j,k] * n)
            if op_result > 255:
                img[i, j, k] = 255
            elif op_result < 0:
                img[i, j, k] = 0
            else:
                img[i, j, k] = <np.uint8_t>op_result