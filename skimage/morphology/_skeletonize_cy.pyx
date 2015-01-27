#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

'''
Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Original author: Lee Kamentsky
'''

import numpy as np
cimport numpy as cnp


def _skeletonize_loop(cnp.uint8_t[:, ::1] result,
                      Py_ssize_t[::1] i, Py_ssize_t[::1] j,
                      cnp.int32_t[::1] order, cnp.uint8_t[::1] table):
    """
    Inner loop of skeletonize function

    Parameters
    ----------

    result : ndarray of uint8
        On input, the image to be skeletonized, on output the skeletonized
        image.

    i, j : ndarrays
        The coordinates of each foreground pixel in the image

    order : ndarray
        The index of each pixel, in the order of processing (order[0] is
        the first pixel to process, etc.)

    table : ndarray
        The 512-element lookup table of values after transformation
        (whether to keep or not each configuration in a binary 3x3 array)

    Notes
    -----

    The loop determines whether each pixel in the image can be removed without
    changing the Euler number of the image. The pixels are ordered by
    increasing distance from the background which means a point nearer to
    the quench-line of the brushfire will be evaluated later than a
    point closer to the edge.

    Note that the neighbourhood of a pixel may evolve before the loop
    arrives at this pixel. This is why it is possible to compute the
    skeleton in only one pass, thanks to an adapted ordering of the
    pixels.
    """
    cdef:
        Py_ssize_t accumulator
        Py_ssize_t index, order_index
        Py_ssize_t ii, jj
        Py_ssize_t rows = result.shape[0]
        Py_ssize_t cols = result.shape[1]

    for index in range(order.shape[0]):
        accumulator = 16
        order_index = order[index]
        ii = i[order_index]
        jj = j[order_index]
        # Compute the configuration around the pixel
        if ii > 0:
            if jj > 0 and result[ii - 1, jj - 1]:
                accumulator += 1
            if result[ii - 1, jj]:
                accumulator += 2
            if jj < cols - 1 and result[ii - 1, jj + 1]:
                    accumulator += 4
            if jj > 0 and result[ii, jj - 1]:
                accumulator += 8
            if jj < cols - 1 and result[ii, jj + 1]:
                accumulator += 32
            if ii < rows - 1:
                if jj > 0 and result[ii + 1, jj - 1]:
                    accumulator += 64
                if result[ii + 1, jj]:
                    accumulator += 128
                if jj < cols - 1 and result[ii + 1, jj + 1]:
                    accumulator += 256
            # Assign the value of table corresponding to the configuration
            result[ii, jj] = table[accumulator]



def _table_lookup_index(cnp.uint8_t[:, ::1] image):
    """
    Return an index into a table per pixel of a binary image

    Take the sum of true neighborhood pixel values where the neighborhood
    looks like this::

       1   2   4
       8  16  32
      64 128 256

    This code could be replaced by a convolution with the kernel::

      256 128 64
       32  16  8
        4   2  1

    but this runs about twice as fast because of inlining and the
    hardwired kernel.
    """
    cdef:
        Py_ssize_t[:, ::1] indexer
        Py_ssize_t *p_indexer
        cnp.uint8_t *p_image
        Py_ssize_t i_stride
        Py_ssize_t i_shape
        Py_ssize_t j_shape
        Py_ssize_t i
        Py_ssize_t j
        Py_ssize_t offset

    i_shape   = image.shape[0]
    j_shape   = image.shape[1]
    indexer = np.zeros((i_shape, j_shape), dtype=np.intp)
    p_indexer = &indexer[0, 0]
    p_image   = &image[0, 0]
    i_stride  = image.strides[0]
    assert i_shape >= 3 and j_shape >= 3, \
        "Please use the slow method for arrays < 3x3"

    for i in range(1, i_shape-1):
        offset = i_stride* i + 1
        for j in range(1, j_shape - 1):
            if p_image[offset]:
                p_indexer[offset + i_stride + 1] += 1
                p_indexer[offset + i_stride] += 2
                p_indexer[offset + i_stride - 1] += 4
                p_indexer[offset + 1] += 8
                p_indexer[offset] += 16
                p_indexer[offset - 1] += 32
                p_indexer[offset - i_stride + 1] += 64
                p_indexer[offset - i_stride] += 128
                p_indexer[offset - i_stride - 1] += 256
            offset += 1
    #
    # Do the corner cases (literally)
    #
    if image[0, 0]:
        indexer[0, 0] += 16
        indexer[0, 1] += 8
        indexer[1, 0] += 2
        indexer[1, 1] += 1

    if image[0, j_shape - 1]:
        indexer[0, j_shape - 2] += 32
        indexer[0, j_shape - 1] += 16
        indexer[1, j_shape - 2] += 4
        indexer[1, j_shape - 1] += 2

    if image[i_shape - 1, 0]:
        indexer[i_shape - 2, 0] += 128
        indexer[i_shape - 2, 1] += 64
        indexer[i_shape - 1, 0] += 16
        indexer[i_shape - 1, 1] += 8

    if image[i_shape - 1, j_shape - 1]:
        indexer[i_shape - 2, j_shape - 2] += 256
        indexer[i_shape - 2, j_shape - 1] += 128
        indexer[i_shape - 1, j_shape - 2] += 32
        indexer[i_shape - 1, j_shape - 1] += 16
    #
    # Do the edges
    #
    for j in range(1, j_shape - 1):
        if image[0, j]:
            indexer[0, j - 1] += 32
            indexer[0, j] += 16
            indexer[0, j + 1] += 8
            indexer[1, j - 1] += 4
            indexer[1, j] += 2
            indexer[1, j + 1] += 1
        if image[i_shape - 1, j]:
            indexer[i_shape - 2, j - 1] += 256
            indexer[i_shape - 2, j] += 128
            indexer[i_shape - 2, j + 1] += 64
            indexer[i_shape - 1, j - 1] += 32
            indexer[i_shape - 1, j] += 16
            indexer[i_shape - 1, j + 1] += 8

    for i in range(1, i_shape - 1):
        if image[i, 0]:
            indexer[i - 1, 0] += 128
            indexer[i, 0] += 16
            indexer[i + 1, 0] += 2
            indexer[i - 1, 1] += 64
            indexer[i, 1] += 8
            indexer[i + 1, 1] += 1
        if image[i, j_shape - 1]:
            indexer[i - 1, j_shape - 2] += 256
            indexer[i, j_shape - 2] += 32
            indexer[i + 1, j_shape - 2] += 4
            indexer[i - 1, j_shape - 1] += 128
            indexer[i, j_shape - 1] += 16
            indexer[i + 1, j_shape - 1] += 2
    return np.asarray(indexer)
