#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp

def _fast_skeletonize(image):
    """Return the skeleton of a binary image.
    Thinning is used to reduce each connected component in a binary image
    to a single-pixel wide skeleton.
    Parameters
    ----------
    image : numpy.ndarray
        A binary image containing the objects to be skeletonized. '1'
        represents foreground, and '0' represents background. It
        also accepts arrays of boolean values where True is foreground.
    Returns
    -------
    skeleton : ndarray
        A matrix containing the thinned image.
    See also
    --------
    medial_axis
    Notes
    -----
    The algorithm [1] works by making successive passes of the image,
    removing pixels on object borders. This continues until no
    more pixels can be removed.  The image is correlated with a
    mask that assigns each pixel a number in the range [0...255]
    corresponding to each possible pattern of its 8 neighbouring
    pixels. A look up table is then used to assign the pixels a
    value of 0, 1, 2 or 3, which are selectively removed during
    the iterations.
    Note that this algorithm will give different results than a
    medial axis transform, which is also often referred to as
    "skeletonization".
    References
    ----------
    .. [1] A fast parallel algorithm for thinning digital patterns,
       T. Y. ZHANG and C. Y. SUEN, Communications of the ACM,
       March 1984, Volume 27, Number 3
    Examples
    --------
    >>> X, Y = np.ogrid[0:9, 0:9]
    >>> ellipse = (1./3 * (X - 4)**2 + (Y - 4)**2 < 3**2).astype(np.uint8)
    >>> ellipse
    array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=uint8)
    >>> skel = skeletonize(ellipse)
    >>> skel.astype(np.uint8)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """

    if image.ndim != 2:
        raise ValueError("Skeletonize requires a 2D array")
    if not np.all(np.in1d(image.flat, (0, 1))):
        raise ValueError("Image contains values other than 0 and 1")

    # look up table - there is one entry for each of the 2^8=256 possible
    # combinations of 8 binary neighbours. 1's, 2's and 3's are candidates
    # for removal at each iteration of the algorithm.
    cdef int *lut = \
      [0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0,
       0, 0, 2, 0, 2, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0,
       0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0, 0, 0, 3, 1,
       0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3, 0, 0,
       1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3,
       0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0]

    cdef int pixel_removed, odd_loop, neighbors

    # indices for fast iteration
    cdef Py_ssize_t x, y

    cdef Py_ssize_t ymax = image.shape[0]+2, xmax = image.shape[1]+2

    # we copy over the image into a larger version with a single pixel border
    # this removes the need to handle border cases below
    _skeleton = np.zeros((ymax, xmax), dtype=np.uint8)
    _skeleton[1:ymax-1, 1:xmax-1] = image > 0

    _cleaned_skeleton = _skeleton.copy()

    # cdef'd numpy-arrays for fast, typed access
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] skeleton, cleaned_skeleton

    skeleton = _skeleton
    cleaned_skeleton = _cleaned_skeleton

    pixel_removed = True

    # the algorithm reiterates the thinning till
    # no further thinning occured (variable pixel_removed set)

    while pixel_removed:
        pixel_removed = False

        # there are two phases, in the first phase, pixels labeled (see below)
        # 1 and 3 are removed, in the second 2 and 3

        for odd_loop in range(1, -1, -1):
            for y in range(1, ymax-1):
                for x in range(1, xmax-1):
                    # all set pixels ...
                    if skeleton[y, x] > 0:
                        # are correlated with a kernel (coefficients spread around here ...)
                        # to apply a unique number to every possible neighborhood ...

                        # which is used with the lut to find the "connectivity type"

                        neighbors = lut[  1*skeleton[y - 1, x - 1] +   2*skeleton[y - 1, x] +\
                                          4*skeleton[y - 1, x + 1] +   8*skeleton[y, x + 1] +\
                                         16*skeleton[y + 1, x + 1] +  32*skeleton[y + 1, x] +\
                                         64*skeleton[y + 1, x - 1] + 128*skeleton[y, x - 1]]

                        # if the condition is met, the pixel is removed (unset)
                        if (odd_loop and neighbors == 1) or ((not odd_loop) and neighbors == 2) or neighbors == 3:
                            cleaned_skeleton[y, x] = 0
                            pixel_removed = True

            # once a step has been processed, the original skeleton
            # is overwritten with the cleaned version
            _skeleton = _cleaned_skeleton.copy()
            skeleton = _skeleton

    return _skeleton[1:ymax-1, 1:xmax-1].astype(np.bool)


"""
Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Original author: Lee Kamentsky
"""

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
