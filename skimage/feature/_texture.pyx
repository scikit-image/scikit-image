#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos, abs
from .._shared.interpolation cimport bilinear_interpolation, round


def _glcm_loop(cnp.uint8_t[:, ::1] image, double[:] distances,
               double[:] angles, Py_ssize_t levels,
               cnp.uint32_t[:, :, :, ::1] out):
    """Perform co-occurrence matrix accumulation.

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
        Py_ssize_t a_idx, d_idx, r, c, rows, cols, row, col
        cnp.uint8_t i, j
        cnp.float64_t angle, distance

    rows = image.shape[0]
    cols = image.shape[1]

    for a_idx in range(len(angles)):
        angle = angles[a_idx]
        for d_idx in range(len(distances)):
            distance = distances[d_idx]
            for r in range(rows):
                for c in range(cols):
                    i = image[r, c]

                    # compute the location of the offset pixel
                    row = r + <int>round(sin(angle) * distance)
                    col = c + <int>round(cos(angle) * distance)

                    # make sure the offset is within bounds
                    if row >= 0 and row < rows and \
                       col >= 0 and col < cols:
                        j = image[row, col]

                        if i >= 0 and i < levels and \
                           j >= 0 and j < levels:
                            out[i, j, d_idx, a_idx] += 1


cdef inline int _bit_rotate_right(int value, int length):
    """Cyclic bit shift to the right.

    Parameters
    ----------
    value : int
        integer value to shift
    length : int
        number of bits of integer

    """
    return (value >> 1) | ((value & 1) << (length - 1))


def _local_binary_pattern(double[:, ::1] image,
                          int P, float R, char method='D'):
    """Gray scale and rotation invariant LBP (Local Binary Patterns).

    LBP is an invariant descriptor that can be used for texture classification.

    Parameters
    ----------
    image : (N, M) double array
        Graylevel image.
    P : int
        Number of circularly symmetric neighbour set points (quantization of
        the angular space).
    R : float
        Radius of circle (spatial resolution of the operator).
    method : {'D', 'R', 'U', 'N', 'V'}
        Method to determine the pattern.

        * 'D': 'default'
        * 'R': 'ror'
        * 'U': 'uniform'
        * 'N': 'nri_uniform'
        * 'V': 'var'

    Returns
    -------
    output : (N, M) array
        LBP image.
    """

    # texture weights
    cdef int[::1] weights = 2 ** np.arange(P, dtype=np.int32)
    # local position of texture elements
    rr = - R * np.sin(2 * np.pi * np.arange(P, dtype=np.double) / P)
    cc = R * np.cos(2 * np.pi * np.arange(P, dtype=np.double) / P)
    cdef double[::1] rp = np.round(rr, 5)
    cdef double[::1] cp = np.round(cc, 5)

    # pre-allocate arrays for computation
    cdef double[::1] texture = np.zeros(P, dtype=np.double)
    cdef signed char[::1] signed_texture = np.zeros(P, dtype=np.int8)
    cdef int[::1] rotation_chain = np.zeros(P, dtype=np.int32)

    output_shape = (image.shape[0], image.shape[1])
    cdef double[:, ::1] output = np.zeros(output_shape, dtype=np.double)

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]

    cdef double lbp
    cdef Py_ssize_t r, c, changes, i
    cdef Py_ssize_t rot_index, n_ones
    cdef cnp.int8_t first_zero, first_one

    # To compute the variance features
    cdef double sum_, var_, texture_i

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            for i in range(P):
                texture[i] = bilinear_interpolation(&image[0, 0], rows, cols,
                                                    r + rp[i], c + cp[i],
                                                    'C', 0)
            # signed / thresholded texture
            for i in range(P):
                if texture[i] - image[r, c] >= 0:
                    signed_texture[i] = 1
                else:
                    signed_texture[i] = 0

            lbp = 0

            # if method == 'var':
            if method == 'V':
                # Compute the variance without passing from numpy.
                # Following the LBP paper, we're taking a biased estimate
                # of the variance (ddof=0)
                sum_ = 0.0
                var_ = 0.0
                for i in range(P):
                    texture_i = texture[i]
                    sum_ += texture_i
                    var_ += texture_i * texture_i
                var_ = (var_ - (sum_ * sum_) / P) / P
                if var_ != 0:
                    lbp = var_
                else:
                    lbp = np.nan
            # if method == 'uniform':
            elif method == 'U' or method == 'N':
                # determine number of 0 - 1 changes
                changes = 0
                for i in range(P - 1):
                    changes += abs(signed_texture[i] - signed_texture[i + 1])
                if method == 'N':
                    # Uniform local binary patterns are defined as patterns
                    # with at most 2 value changes (from 0 to 1 or from 1 to
                    # 0). Uniform patterns can be caraterized by their number
                    # `n_ones` of 1.  The possible values for `n_ones` range
                    # from 0 to P.
                    # Here is an example for P = 4:
                    # n_ones=0: 0000
                    # n_ones=1: 0001, 1000, 0100, 0010
                    # n_ones=2: 0011, 1001, 1100, 0110
                    # n_ones=3: 0111, 1011, 1101, 1110
                    # n_ones=4: 1111
                    #
                    # For a pattern of size P there are 2 constant patterns
                    # corresponding to n_ones=0 and n_ones=P. For each other
                    # value of `n_ones` , i.e n_ones=[1..P-1], there are P
                    # possible patterns which are related to each other through
                    # circular permutations. The total number of uniform
                    # patterns is thus (2 + P * (P - 1)).
                    # Given any pattern (uniform or not) we must be able to
                    # associate a unique code:
                    # 1. Constant patterns patterns (with n_ones=0 and
                    #    n_ones=P) and non uniform patterns are given fixed
                    #    code values.
                    # 2. Other uniform patterns are indexed considering the
                    #    value of n_ones, and an index called 'rot_index'
                    #    reprenting the number of circular right shifts
                    #    required to obtain the pattern starting from a
                    #    reference position (corresponding to all zeros stacked
                    #    on the right). This number of rotations (or circular
                    #    right shifts) 'rot_index' is efficiently computed by
                    #    considering the positions of the first 1 and the first
                    #    0 found in the pattern.

                    if changes <= 2:
                        # We have a uniform pattern
                        n_ones = 0  # determines the number of ones
                        first_one = -1  # position was the first one
                        first_zero = -1  # position of the first zero
                        for i in range(P):
                            if signed_texture[i]:
                                n_ones += 1
                                if first_one == -1:
                                    first_one = i
                            else:
                                if first_zero == -1:
                                    first_zero = i
                        if n_ones == 0:
                            lbp = 0
                        elif n_ones == P:
                            lbp = P * (P - 1) + 1
                        else:
                            if first_one == 0:
                                rot_index = n_ones - first_zero
                            else:
                                rot_index = P - first_one
                            lbp = 1 + (n_ones - 1) * P + rot_index
                    else:  # changes > 2
                        lbp = P * (P - 1) + 2
                else:  # method != 'N'
                    if changes <= 2:
                        for i in range(P):
                            lbp += signed_texture[i]
                    else:
                        lbp = P + 1
            else:
                # method == 'default'
                for i in range(P):
                    lbp += signed_texture[i] * weights[i]

                # method == 'ror'
                if method == 'R':
                    # shift LBP P times to the right and get minimum value
                    rotation_chain[0] = <int>lbp
                    for i in range(1, P):
                        rotation_chain[i] = \
                            _bit_rotate_right(rotation_chain[i - 1], P)
                    lbp = rotation_chain[0]
                    for i in range(1, P):
                        lbp = min(lbp, rotation_chain[i])

            output[r, c] = lbp

    return np.asarray(output)
