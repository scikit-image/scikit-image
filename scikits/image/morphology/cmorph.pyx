"""
:author: Damian Eads, 2009
:license: modified BSD
"""

from __future__ import division
import numpy as np

cimport numpy as np
cimport cython
from cpython cimport bool

STREL_DTYPE = np.uint8
ctypedef np.uint8_t STREL_DTYPE_t

IMAGE_DTYPE = np.uint8
ctypedef np.uint8_t IMAGE_DTYPE_t

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

@cython.boundscheck(False)
def dilate(np.ndarray[IMAGE_DTYPE_t, ndim=2] image not None,
           np.ndarray[IMAGE_DTYPE_t, ndim=2] selem not None,
           np.ndarray[IMAGE_DTYPE_t, ndim=2] out,
           bool shift_x, bool shift_y):
    cdef int hw = selem.shape[0] // 2
    cdef int hh = selem.shape[1] // 2
    if shift_x:
        hh -= 1
    if shift_y:
        hw -= 1

    cdef int width = image.shape[0], height = image.shape[1]
    if out is None:
        out = np.zeros([width, height], dtype=IMAGE_DTYPE)

    assert out.shape[0] == image.shape[0]
    assert out.shape[1] == image.shape[1]

    cdef int x, y, ix, iy, cx, cy
    cdef IMAGE_DTYPE_t max_so_far

    cdef int sw = selem.shape[0], sh = selem.shape[1]

    cdef np.ndarray[np.int_t, ndim=2] xinc = np.zeros([sw, sh], dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=2] yinc = np.zeros([sw, sh], dtype=np.int)

    for x in range(sw):
        for y in range(sh):
            xinc[x, y] = (x - hw)
            yinc[x, y] = (y - hh)


    for x in range(width):
        for y in range(height):
            max_so_far = 0
            for cx in range(0, sw):
                for cy in range(0, sh):
                    ix = x + xinc[cx,cy]
                    iy = y + yinc[cx,cy]
                    if ix>=0 and iy>=0 and ix < width and iy < height \
                           and selem[cx, cy] == 1 \
                           and image[ix,iy] > max_so_far:
                        max_so_far = image[ix,iy]
            out[x,y] = max_so_far

    return out


@cython.boundscheck(False)
def erode(np.ndarray[IMAGE_DTYPE_t, ndim=2] image not None,
          np.ndarray[IMAGE_DTYPE_t, ndim=2] selem not None,
          np.ndarray[IMAGE_DTYPE_t, ndim=2] out,
          bool shift_x, bool shift_y):
    cdef int hw = selem.shape[0] // 2
    cdef int hh = selem.shape[1] // 2
    if shift_x:
        hh -= 1
    if shift_y:
        hw -= 1

    cdef int width = image.shape[0], height = image.shape[1]
    if out is None:
        out = np.zeros([width, height], dtype=IMAGE_DTYPE)

    assert out.shape[0] == image.shape[0]
    assert out.shape[1] == image.shape[1]

    cdef int x, y, ix, iy, cx, cy
    cdef IMAGE_DTYPE_t min_so_far

    cdef int sw = selem.shape[0], sh = selem.shape[1]

    cdef np.ndarray[np.int_t, ndim=2] xinc = np.zeros([sw, sh], dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=2] yinc = np.zeros([sw, sh], dtype=np.int)

    for x in range(sw):
        for y in range(sh):
            xinc[x, y] = (x - hw)
            yinc[x, y] = (y - hh)

    for x in range(width):
        for y in range(height):
            min_so_far = 255
            for cx in range(0, sw):
                for cy in range(0, sh):
                    ix = x + xinc[cx,cy]
                    iy = y + yinc[cx,cy]
                    if ix>=0 and iy>=0 and ix < width \
                           and iy < height and selem[cx, cy] == 1 \
                           and image[ix,iy] < min_so_far:
                        min_so_far = image[ix,iy]
            out[x,y] = min_so_far

    return out
