import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, cos, abs, ceil, floor


@cython.boundscheck(False)
def _glcm_loop(np.ndarray[dtype=np.uint8_t, ndim=2,
                          negative_indices=False, mode='c'] image,
               np.ndarray[dtype=np.float64_t, ndim=1,
                          negative_indices=False, mode='c'] distances,
               np.ndarray[dtype=np.float64_t, ndim=1,
                          negative_indices=False, mode='c'] angles,
               int levels,
               np.ndarray[dtype=np.uint32_t, ndim=4,
                          negative_indices=False, mode='c'] out
               ):
    """Perform co-occurnace matrix accumulation

    Parameters
    ----------
    image : ndarray
        Input image, which is converted to the uint8 data type.
    distances : ndarray
        List of pixel pair distance offsets.
    angles : ndarray
        List of pixel pair angles in radians.
    levels : int
        The input image should contain integers in [0, levels-1],
        where levels indicate the number of grey-levels counted
        (typically 256 for an 8-bit image)
    out : ndarray
        On input a 4D array of zeros, and on output it contains
        the results of the GLCM computation.

    """
    cdef:
        np.int32_t a_inx, d_idx
        np.int32_t r, c, rows, cols, row, col
        np.int32_t i, j

    rows = image.shape[0]
    cols = image.shape[1]

    for a_idx, angle in enumerate(angles):
        for d_idx, distance in enumerate(distances):
            for r in range(rows):
                for c in range(cols):
                    i = image[r, c]

                    # compute the location of the offset pixel
                    row = r + <int>(sin(angle) * distance + 0.5)
                    col = c + <int>(cos(angle) * distance + 0.5);

                    # make sure the offset is within bounds
                    if row >= 0 and row < rows and \
                       col >= 0 and col < cols:
                        j = image[row, col]

                        if i >= 0 and i < levels and \
                           j >= 0 and j < levels:
                            out[i, j, d_idx, a_idx] += 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef _bilinear_interpolation(np.ndarray[double, ndim=2] image,
                             np.ndarray[double, ndim=2] coords,
                             np.ndarray[double, ndim=1] output,
                             double r0=0, double c0=0, double cval=0):
    cdef double r, c, dr, dc
    cdef int i, minr, minc, maxr, maxc

    for i in range(coords.shape[0]):
        r = r0 + coords[i, 0]
        c = c0 + coords[i, 1]
        minr = <int>floor(r)
        minc = <int>floor(c)
        maxr = <int>ceil(r)
        maxc = <int>ceil(c)
        dr = r - minr
        dc = c - minc
        if (
            minr < 0 or maxr >= image.shape[0]
            or minc < 0 or maxc >= image.shape[1]
        ):
            output[i] = cval
        else:
            top = (1 - dc) * image[minr, minc] + dc * image[minr, maxc]
            bottom = (1 - dc) * image[maxr, minc] + dc * image[maxr, maxc]
            output[i] = (1 - dr) * top + dr * bottom


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int _bit_rotate_right(int value, int length):
    """Cyclic bit shift to the right.

    Parameters
    ----------
    value : int
        integer value to shift
    length : int
        number of bits of integer

    """
    return (value >> 1) | ((value & 1) << (length - 1))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _local_binary_pattern(np.ndarray[double, ndim=2] image,
                          int P, float R, int method=0):
    # texture weights
    cdef np.ndarray[int, ndim=1] weights = 2 ** np.arange(P, dtype='int32')
    # local position of texture elements
    rp = - R * np.sin(2 * np.pi * np.arange(P, dtype='double') / P)
    cp = R * np.cos(2 * np.pi * np.arange(P, dtype='double') / P)
    cdef np.ndarray[double, ndim=2] coords = np.round(np.vstack([rp, cp]).T, 5)

    # pre allocate arrays for computation
    cdef np.ndarray[double, ndim=1] texture = np.zeros(P, 'double')
    cdef np.ndarray[char, ndim=1] signed_texture = np.zeros(P, 'int8')
    cdef np.ndarray[int, ndim=1] rotation_chain = np.zeros(P, 'int32')

    output_shape = (image.shape[0], image.shape[1])
    cdef np.ndarray[double, ndim=2] output = np.zeros(output_shape, 'double')

    cdef double lbp
    cdef int r, c, changes, i
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            _bilinear_interpolation(image, coords, texture, r, c)
            # signed / thresholded texture
            for i in range(P):
                if texture[i] - image[r, c] >= 0:
                    signed_texture[i] = 1
                else:
                    signed_texture[i] = 0

            lbp = 0

            # if method == 'uniform' or method == 'var':
            if method == 2 or method == 3:
                # determine number of 0 - 1 changes
                changes = 0
                for i in range(P - 1):
                    changes += abs(signed_texture[i] - signed_texture[i + 1])

                if changes <= 2:
                    for i in range(P):
                        lbp += signed_texture[i]
                else:
                    lbp = P + 1

                if method == 3:
                    var = np.var(texture)
                    if var != 0:
                        lbp /= var
                    else:
                        lbp = np.nan
            else:
                # method == 'default'
                for i in range(P):
                    lbp += signed_texture[i] * weights[i]

                if method == 1:
                    # shift LBP P times to the right and get minimum value
                    rotation_chain[0] = <int>lbp
                    for i in range(1, P):
                        rotation_chain[i] = \
                            _bit_rotate_right(rotation_chain[i - 1], P)
                    lbp = rotation_chain[0]
                    for i in range(1, P):
                        lbp = min(lbp, rotation_chain[i])

            output[r, c] = lbp

    return output
