#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos, abs
from .._shared.interpolation cimport bilinear_interpolation, round
from .._shared.transform cimport integrate

cdef extern from "numpy/npy_math.h":
    double NAN "NPY_NAN"

from .._shared.fused_numerics cimport np_anyint as any_int
from .._shared.fused_numerics cimport np_real_numeric

cnp.import_array()

def _glcm_loop(any_int[:, ::1] image, double[:] distances,
               double[:] angles, Py_ssize_t levels,
               cnp.uint32_t[:, :, :, ::1] out):
    """Perform co-occurrence matrix accumulation.

    Parameters
    ----------
    image : ndarray
        Integer typed input image. Only positive valued images are supported.
        If type is other than uint8, the argument `levels` needs to be set.
    distances : ndarray
        List of pixel pair distance offsets.
    angles : ndarray
        List of pixel pair angles in radians.
    levels : int
        The input image should contain integers in [0, `levels`-1],
        where levels indicate the number of gray-levels counted
        (typically 256 for an 8-bit image).
    out : ndarray
        On input a 4D array of zeros, and on output it contains
        the results of the GLCM computation.

    """

    cdef:
        Py_ssize_t a_idx, d_idx, r, c, rows, cols, row, col, start_row,\
                   end_row, start_col, end_col, offset_row, offset_col
        any_int i, j
        cnp.float64_t angle, distance

    with nogil:
        rows = image.shape[0]
        cols = image.shape[1]

        for a_idx in range(angles.shape[0]):
            angle = angles[a_idx]
            for d_idx in range(distances.shape[0]):
                distance = distances[d_idx]
                offset_row = round(sin(angle) * distance)
                offset_col = round(cos(angle) * distance)
                start_row = max(0, -offset_row)
                end_row = min(rows, rows - offset_row)
                start_col = max(0, -offset_col)
                end_col = min(cols, cols - offset_col)
                for r in range(start_row, end_row):
                    for c in range(start_col, end_col):
                        i = image[r, c]
                        # compute the location of the offset pixel
                        row = r + offset_row
                        col = c + offset_col
                        j = image[row, col]
                        if 0 <= i < levels and 0 <= j < levels:
                            out[i, j, d_idx, a_idx] += 1


cdef inline int _bit_rotate_right(int value, int length) nogil:
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
                          int P, float R, char method=b'D'):
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

    with nogil:
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                for i in range(P):
                    bilinear_interpolation[cnp.float64_t, double, double](
                            &image[0, 0], rows, cols, r + rp[i], c + cp[i],
                            b'C', 0, &texture[i])
                # signed / thresholded texture
                for i in range(P):
                    if texture[i] - image[r, c] >= 0:
                        signed_texture[i] = 1
                    else:
                        signed_texture[i] = 0

                lbp = 0

                # if method == b'var':
                if method == b'V':
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
                        lbp = NAN
                # if method == b'uniform':
                elif method == b'U' or method == b'N':
                    # determine number of 0 - 1 changes
                    changes = 0
                    for i in range(P - 1):
                        changes += (signed_texture[i]
                                    - signed_texture[i + 1]) != 0
                    if method == b'N':
                        # Uniform local binary patterns are defined as patterns
                        # with at most 2 value changes (from 0 to 1 or from 1 to
                        # 0). Uniform patterns can be characterized by their
                        # number `n_ones` of 1.  The possible values for
                        # `n_ones` range from 0 to P.
                        #
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
                        # possible patterns which are related to each other
                        # through circular permutations. The total number of
                        # uniform patterns is thus (2 + P * (P - 1)).

                        # Given any pattern (uniform or not) we must be able to
                        # associate a unique code:
                        #
                        # 1. Constant patterns patterns (with n_ones=0 and
                        # n_ones=P) and non uniform patterns are given fixed
                        # code values.
                        #
                        # 2. Other uniform patterns are indexed considering the
                        # value of n_ones, and an index called 'rot_index'
                        # reprenting the number of circular right shifts
                        # required to obtain the pattern starting from a
                        # reference position (corresponding to all zeros stacked
                        # on the right). This number of rotations (or circular
                        # right shifts) 'rot_index' is efficiently computed by
                        # considering the positions of the first 1 and the first
                        # 0 found in the pattern.

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
                    # method == b'default'
                    for i in range(P):
                        lbp += signed_texture[i] * weights[i]

                    # method == b'ror'
                    if method == b'R':
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


# Constant values that are used by `_multiblock_lbp` function.
# Values represent offsets of neighbour rectangles relative to central one.
# It has order starting from top left and going clockwise.
cdef:
    Py_ssize_t[::1] mlbp_r_offsets = np.asarray([-1, -1, -1, 0, 1, 1, 1, 0],
                                                dtype=np.intp)
    Py_ssize_t[::1] mlbp_c_offsets = np.asarray([-1, 0, 1, 1, 1, 0, -1, -1],
                                                dtype=np.intp)


cpdef int _multiblock_lbp(np_floats[:, ::1] int_image,
                          Py_ssize_t r,
                          Py_ssize_t c,
                          Py_ssize_t width,
                          Py_ssize_t height) nogil:
    """Multi-block local binary pattern (MB-LBP) [1]_.

    Parameters
    ----------
    int_image : (N, M) float array
        Integral image.
    r : int
        Row-coordinate of top left corner of a rectangle containing feature.
    c : int
        Column-coordinate of top left corner of a rectangle containing feature.
    width : int
        Width of one of 9 equal rectangles that will be used to compute
        a feature.
    height : int
        Height of one of 9 equal rectangles that will be used to compute
        a feature.

    Returns
    -------
    output : int
        8-bit MB-LBP feature descriptor.

    References
    ----------
    .. [1] L. Zhang, R. Chu, S. Xiang, S. Liao, S.Z. Li. "Face Detection Based
           on Multi-Block LBP Representation", In Proceedings: Advances in
           Biometrics, International Conference, ICB 2007, Seoul, Korea.
           http://www.cbsr.ia.ac.cn/users/scliao/papers/Zhang-ICB07-MBLBP.pdf
           :DOI:`10.1007/978-3-540-74549-5_2`
    """

    cdef:
        # Top-left coordinates of central rectangle.
        Py_ssize_t central_rect_r = r + height
        Py_ssize_t central_rect_c = c + width

        Py_ssize_t r_shift = height - 1
        Py_ssize_t c_shift = width - 1

        Py_ssize_t current_rect_r, current_rect_c
        Py_ssize_t element_num, i
        np_floats current_rect_val
        int has_greater_value
        int lbp_code = 0

    # Sum of intensity values of central rectangle.
    cdef float central_rect_val = integrate(int_image, central_rect_r,
                                            central_rect_c,
                                            central_rect_r + r_shift,
                                            central_rect_c + c_shift)

    for element_num in range(8):

        current_rect_r = central_rect_r + mlbp_r_offsets[element_num]*height
        current_rect_c = central_rect_c + mlbp_c_offsets[element_num]*width


        current_rect_val = integrate(int_image, current_rect_r, current_rect_c,
                                     current_rect_r + r_shift,
                                     current_rect_c + c_shift)


        has_greater_value = current_rect_val >= central_rect_val

        # If current rectangle's intensity value is bigger
        # make corresponding bit to 1.
        lbp_code |= has_greater_value << (7 - element_num)

    return lbp_code
