# -*- python -*-

"""Colour Mixer

NumPy does not do overflow checking when adding or multiplying
integers, so currently the only way to clip results efficiently
(without making copies of the data) is with an extension such as this
one.

"""

import numpy as np
cimport numpy as np

import cython

@cython.boundscheck(False)
def add(np.ndarray[np.uint8_t, ndim=3] img,
        np.ndarray[np.uint8_t, ndim=3] stateimg,
        int channel, int amount):
    """Add a given amount to a colour channel of `stateimg`, and
    store the result in `img`.  Overflow is clipped.

    Parameters
    ----------
    img : (M, N, 3) ndarray of uint8
        Output image.
    stateimg : (M, N, 3) ndarray of uint8
        Input image.
    channel : int
        Channel (0 for "red", 1 for "green", 2 for "blue").
    amount : int
        Value to add.

    """
    cdef int height = img.shape[0]
    cdef int width = img.shape[1]
    cdef int k = channel
    cdef int n = amount

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
             int channel, float amount):
    """Multiply a colour channel of `stateimg` by a certain amount, and
    store the result in `img`.  Overflow is clipped.

    Parameters
    ----------
    img : (M, N, 3) ndarray of uint8
        Output image.
    stateimg : (M, N, 3) ndarray of uint8
        Input image.
    channel : int
        Channel (0 for "red", 1 for "green", 2 for "blue").
    amount : float
        Multiplication factor.

    """
    cdef int height = img.shape[0]
    cdef int width = img.shape[1]
    cdef int k = channel
    cdef float n = amount

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
