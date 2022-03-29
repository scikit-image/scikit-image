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

ctypedef fused uint32_or_float64:
    cnp.uint32_t
    cnp.float64_t

ctypedef fused any_int_or_float:
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.float32_t
    cnp.float64_t

cnp.import_array()


def _glcm_loop(any_int[:, ::1] image, double[:] distances,
               double[:] angles, Py_ssize_t levels,
               uint32_or_float64[:, :, :, ::1] out,
               uint32_or_float64[:, ::1] out_total,
               bint symmetric,
               bint normed):
    """Perform co-occurrence matrix accumulation.

    Cython helper / computation function for `greycomatrix`

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
        On input a 4D array of zeros.
        On output it contains the grey level co-occurrence matrix for `image`.
        See `greycomatrix` for more details. This routine does most of the
        numerical computation for `greycomatrix`.
    out_total: ndarray
        On input a 2D array of zeros.
        On output, if `normed` is True, it contains the totals used to normalise
        for each given offset. If `normed` is False, it will still contain zeros.
    symmetric : boolean
        If True, the output matrix `P[:, :, d, theta]` is symmetric.
    normed:
        If True, normalize `out` by dividing by the total number of
        accumulated co-occurrences for the given offset. The elements
        of the resulting matrix sum to 1.
    """

    cdef:
        Py_ssize_t a_idx, d_idx, r, c, rows, cols, row, col, start_row,\
                   end_row, start_col, end_col, offset_row, offset_col,\
                   idx, jdx
        any_int i, j
        cnp.float64_t angle, distance
        uint32_or_float64 total_increment = 2 if symmetric else 1
        uint32_or_float64 symmetric_increment = 1 if symmetric else 0
    num_dist = distances.shape[0]
    num_angle = angles.shape[0]

    with nogil:
        rows = image.shape[0]
        cols = image.shape[1]

        for a_idx in range(num_angle):
            angle = angles[a_idx]
            for d_idx in range(num_dist):
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
                            out[j, i, d_idx, a_idx] += symmetric_increment
                            if normed:
                                out_total[d_idx, a_idx] += total_increment
        if normed:
            # normalize each GLCM
            for d_idx in range(num_dist):
                for a_idx in range(num_angle):
                    if out_total[d_idx, a_idx] != 0:
                        for idx in range(levels):
                            for jdx in range(levels):
                                out[idx, jdx, d_idx, a_idx] /= out_total[d_idx, a_idx]


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
        Number of circularly symmetric neighbor set points (quantization of
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
# Values represent offsets of neighbor rectangles relative to central one.
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


def _glcm_norm(double[:, :, :, ::1] P,
               Py_ssize_t num_level,
               Py_ssize_t num_dist,
               Py_ssize_t num_angle):
    """Perform co-occurrence matrix normalisation.

    Parameters
    ----------
    P : float64 array
        Input array. `P` is the grey-level co-occurrence histogram
        for which to compute the specified property. The value
        `P[i, j, d, theta]` is the number of times that grey-level j
        occurs at a distance d and at an angle theta from
        grey-level i.
    num_level: int
    num_dist: int
    num_angle: int
        Input array P should be of shape
        [num_level, num_level, num_dist, num_angle]
    """

    cdef:
        Py_ssize_t a_idx, d_idx, idx, jdx
        cnp.float64_t acc

    with nogil:
        # normalize each GLCM
        for d_idx in range(num_dist):
            for a_idx in range(num_angle):
                acc = 0.0
                for idx in range(num_level):
                    for jdx in range(num_level):
                        acc += P[idx, jdx, d_idx, a_idx]
                if acc != 0.0:
                    for idx in range(num_level):
                        for jdx in range(num_level):
                            P[idx, jdx, d_idx, a_idx] /= acc


def _coprop_weights(cnp.float64_t[:, :, :, ::1] P,
                    Py_ssize_t num_level,
                    Py_ssize_t num_dist,
                    Py_ssize_t num_angle,
                    bint pre_normalized,
                    str prop):
    """Calculate property matrix of a GLCM for certain properties.

    For `prop` in ['contrast', 'dissimilarity', 'homogeneity'], calculate
    property of GLCM matrix P. Cython helper function for greycoprops.

    Parameters
    ----------
    P : ndarray
        Grey-level co-occurrence matrix, typically as output from `greycomatrix`.
    num_level : int
        The number of grey levels in the original image.
    num_dist : int
        Number of pixel pair distance offsets.
    num_angle : int
        Number of pixel pair angles.
    pre_normalized : boolean
        Flag if `P` is *already* normalised. Mirrors `normed` flag for `greycomatrix`.
    prop : str
        `prop` value from `greycoprops`. Should be one of 'contrast', 'dissimilarity' or 'homogeneity'.

    Returns
    -------
    output : (num_dist, num_angle) ndarray
        2-dimensional array. `results[d, a]` is the property 'prop' for
        the d'th distance and the a'th angle.
    """

    cdef Py_ssize_t idx, jdx, a_idx, d_idx
    cdef cnp.float64_t acc, weight, diff
    cdef bint get_out
    # The line below will cause an error if `prop` is inappropriate
    cdef int i_prop = ['contrast', 'dissimilarity', 'homogeneity'].index(prop)
    out = np.zeros((num_dist, num_angle), dtype=np.float64)
    cdef double[:, ::1] out_view = out
    weights = np.zeros((num_level, num_level), dtype=np.float64)
    cdef double[:, ::1] weights_view = weights

    with nogil:

        for idx in range(num_level):
            for jdx in range(num_level):
                diff = idx - jdx
                if i_prop == 2:
                    # Reciprocal of actual weight to avoid an unnecessary extra division below
                    weights_view[idx, jdx] = 1. + (diff * diff)
                elif i_prop == 0:
                    weights_view[idx, jdx] = diff * diff
                else:
                    weights_view[idx, jdx] = diff if diff > 0 else -diff
        for d_idx in range(num_dist):
            for a_idx in range(num_angle):

                # Accumulate sum of elements of `P` for this `d_idx` / `a_idx`.
                # If `pre_normalized` is True, acc is used only as a flag to indicate whether
                # the weights need applying for this `d_idx` / `a_idx`, hence if / break blocks.
                with gil:
                    acc = _sum_for_all_levels(P, num_level, d_idx, a_idx, pre_normalized)

                # If acc is zero, every value in P for current d_idx and a_idx is also zero, so there's
                # nothing more to do. This assumes that every value in P is zero or positive.
                # We could test for the latter, but it would slow things down
                if acc != 0.0:
                    if i_prop == 2: # homogeneity
                        with gil:
                            out_view[d_idx, a_idx] = _coprop_homogeneity(
                                P, weights_view, num_level, d_idx, a_idx, acc, pre_normalized)
                    else: # contrast or dissimilarity
                        with gil:
                            out_view[d_idx, a_idx] = _coprop_contrast_dissimilarity(
                                P, weights_view, num_level, d_idx, a_idx, acc, pre_normalized)

#                         if pre_normalized: # P is already normalised
#                             for idx in range(num_level):
#                                 for jdx in range(num_level):
#                                     out_view[d_idx, a_idx] += P[idx, jdx, d_idx, a_idx] / weights_view[idx, jdx]
#                         else: # P not normalised
#                             for idx in range(num_level):
#                                 for jdx in range(num_level):
#                                     out_view[d_idx, a_idx] += P[idx, jdx, d_idx, a_idx] / (acc * weights_view[idx, jdx])
#                    else: # contrast or dissimilarity
#                        if pre_normalized: # P is already normalised
#                            for idx in range(num_level):
#                                for jdx in range(num_level):
#                                    out_view[d_idx, a_idx] += P[idx, jdx, d_idx, a_idx] * weights_view[idx, jdx]
#                        else: # P not normalised
#                            for idx in range(num_level):
#                                for jdx in range(num_level):
#                                    out_view[d_idx, a_idx] += P[idx, jdx, d_idx, a_idx] * weights_view[idx, jdx] / acc
    return out


def _sum_for_all_levels(cnp.float64_t[:, :, :, ::1] P,
                        Py_ssize_t num_level,
                        Py_ssize_t d_idx,
                        Py_ssize_t a_idx,
                        bint pre_normalized):
    """Accumulator helper function for _coprop_weights

    Accumulate sum of elements of `P` for given `d_idx` and `a_idx`.
    If `pre_normalized` is True, acc is used only as a flag to indicate
    whether weights need applying for this `d_idx` an `a_idx` in
    _coprop_weights.

    Parameters
    ----------
    P : ndarray
        Grey-level co-occurrence matrix, typically as output from `greycomatrix`.
    num_level : int
        The number of grey levels in the original image.
    d_idx : int
        Index of pixel pair distance offset to be considered.
    a_idx : int
        Index of pixel pair angle to be considered.
    pre_normalized : boolean
        Flag if `P` is *already* normalised. Mirrors `normed` flag for `greycomatrix`.
    """

    cdef Py_ssize_t idx, jdx
    cdef cnp.float64_t acc
    cdef bint finished

    with nogil:
        acc = 0.0
        for idx in range(num_level):
            for jdx in range(num_level):
                acc += P[idx, jdx, d_idx, a_idx]
                if pre_normalized and acc != 0:
                    with gil:
                        return acc
    return acc

def _coprop_homogeneity(cnp.float64_t[:, :, :, ::1] P,
                        double[:, ::1] weights_view,
                        Py_ssize_t num_level,
                        Py_ssize_t d_idx,
                        Py_ssize_t a_idx,
                        cnp.float64_t acc,
                        bint pre_normalized):

    cdef Py_ssize_t idx, jdx
    cdef cnp.float64_t out

    with nogil:
        out = 0.0
        if pre_normalized: # P is already normalised
            for idx in range(num_level):
                for jdx in range(num_level):
                    out += P[idx, jdx, d_idx, a_idx] / weights_view[idx, jdx]
        else: # P not normalised
            for idx in range(num_level):
                for jdx in range(num_level):
                    out += P[idx, jdx, d_idx, a_idx] / (acc * weights_view[idx, jdx])
    return out



def _coprop_contrast_dissimilarity(cnp.float64_t[:, :, :, ::1] P,
                        double[:, ::1] weights_view,
                        Py_ssize_t num_level,
                        Py_ssize_t d_idx,
                        Py_ssize_t a_idx,
                        cnp.float64_t acc,
                        bint pre_normalized):

    cdef Py_ssize_t idx, jdx
    cdef cnp.float64_t out

    with nogil:
        out = 0.0
        if pre_normalized: # P is already normalised
            for idx in range(num_level):
                for jdx in range(num_level):
                    out += P[idx, jdx, d_idx, a_idx] * weights_view[idx, jdx]
        else: # P not normalised
            for idx in range(num_level):
                for jdx in range(num_level):
                    out += P[idx, jdx, d_idx, a_idx] * weights_view[idx, jdx] / acc
    return out
