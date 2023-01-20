#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
from libc.float cimport DBL_MAX
from libc.math cimport M_PI, atan, atan2, cos, fabs, fmax, sqrt

from .._shared.fused_numerics cimport np_floats
from ..util import img_as_float64

cnp.import_array()

def _real_symmetric_2x2_evs(
    np_floats[:, ::1] M00, np_floats[:, ::1] M01, np_floats[:, ::1] M11,
    bint ascending=False, bint abs_sort=False,
):
    """Analytical eigenvalues of a symmetric 2 x 2 matrix."""
    # use double-precision intermediate variables for accuracy
    cdef:
        cnp.float64_t m00, m01, m11, tmp1, tmp2, lam1, lam2, stmp;
        Py_ssize_t rows = M00.shape[0];
        Py_ssize_t cols = M00.shape[1];
        Py_ssize_t r, c;

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef np_floats[:, :, ::1] evals = np.empty((2, rows, cols), dtype=dtype)

    with nogil:
        for r in range(rows):
            for c in range(cols):
                m00 = <cnp.float64_t>(M00[r, c])
                m01 = <cnp.float64_t>(M01[r, c])
                m11 = <cnp.float64_t>(M11[r, c])
                tmp1 = 4 * m01 * m01

                tmp2 = m00 - m11
                tmp2 *= tmp2
                tmp2 += tmp1
                tmp2 = sqrt(tmp2) / 2

                tmp1 = (m00 + m11) / 2
                if ascending:
                    lam1 = tmp1 - tmp2
                    lam2 = tmp1 + tmp2
                    if abs_sort and (fabs(lam1) > fabs(lam2)):
                        stmp = lam1
                        lam1 = lam2
                        lam2 = stmp
                else:
                    lam1 = tmp1 + tmp2
                    lam2 = tmp1 - tmp2
                    if abs_sort and (fabs(lam1) < fabs(lam2)):
                        stmp = lam1
                        lam1 = lam2
                        lam2 = stmp
                evals[0, r, c] = lam1
                evals[1, r, c] = lam2
    return np.asarray(evals)


def _real_symmetric_3x3_evs(
    np_floats[:, :, ::1] M00, np_floats[:, :, ::1] M01, np_floats[:, :, ::1] M02,
    np_floats[:, :, ::1] M11, np_floats[:, :, ::1] M12, np_floats[:, :, ::1] M22,
    bint ascending=False, bint abs_sort=False,
):
    """Analytical eigenvalues of a symmetric 3 x 3 matrix.

    Follows the expressions given for hermitian symmetric 3 x 3 matrices in
    [1]_, but simplified to handle real-valued matrices only.

    Parameters
    ----------
    M00, M01, M02, M11, M12, M22 : cp.ndarray
        Images corresponding to the individual components of the symmteric,
        real-valued matrix M. `M01`, for instance, represents entry ``M[0, 1]``
        (equivalent to ``M[1, 0]`` by symmetry).
    sort : {"ascending", "descending"}, optional
        Eigenvalues should be sorted in the specified order.
    abs_sort : boolean, optional
        If ``True``, sort based on the absolute values.

    References
    ----------
    .. [1] C. Deledalle, L. Denis, S. Tabti, F. Tupin. Closed-form expressions
        of the eigen decomposition of 2 x 2 and 3 x 3 Hermitian matrices.
        [Research Report] Université de Lyon. 2017.
        https://hal.archives-ouvertes.fr/hal-01501221/file/matrix_exp_and_log_formula.pdf
    """  # noqa
    cdef:
        # use double-precision intermediate variables for accuracy
        cnp.float64_t a, b, c, d, e, f, d_sq, e_sq, f_sq;
        cnp.float64_t x1, x2, phi, tmpa, tmpb, tmpc, arg, x1_term, abc;
        cnp.float64_t lam1, lam2, lam3, stmp;
        cnp.float64_t abs_lam1, abs_lam2, abs_lam3;
        Py_ssize_t planes = M00.shape[0];
        Py_ssize_t rows = M00.shape[1];
        Py_ssize_t cols = M00.shape[2];
        Py_ssize_t pln, row, col;

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef np_floats[:, :, :, ::1] evals = np.empty((3, planes, rows, cols),
                                                  dtype=dtype)

    with nogil:
        for pln in range(planes):
            for row in range(rows):
                for col in range(cols):
                    # M = [[a, d, f],
                    #      [d, b, e],
                    #      [f, e, c]]
                    # so d = M01, etc.
                    a = <cnp.float64_t>(M00[pln, row, col])
                    b = <cnp.float64_t>(M11[pln, row, col])
                    c = <cnp.float64_t>(M22[pln, row, col])
                    d = <cnp.float64_t>(M01[pln, row, col])
                    e = <cnp.float64_t>(M12[pln, row, col])
                    f = <cnp.float64_t>(M02[pln, row, col])
                    d_sq = d * d
                    e_sq = e * e
                    f_sq = f * f
                    tmpa = 2 * a - b - c
                    tmpb = 2 * b - a - c
                    tmpc = 2 * c - a - b
                    x2 = - tmpa * tmpb * tmpc
                    x2 += 9 * (tmpc * d_sq + tmpb * f_sq + tmpa * e_sq)
                    x2 -= 54 * d * e * f
                    x1 = a*a + b*b + c*c - a*b - a*c - b*c + 3 * (d_sq + e_sq + f_sq)
                    x1 = fmax(x1, 0.0)

                    if x2 == 0.0:
                        phi = M_PI / 2.0
                    else:
                        # added max() here for numerical stability
                        # (avoid NaN values in test_hessian_matrix_eigvals_3d)
                        arg = fmax(4*x1*x1*x1 - x2*x2, 0.0)
                        phi = atan(sqrt(arg)/x2)
                        if x2 < 0:
                            phi += M_PI
                    x1_term = (2.0 / 3.0) * sqrt(x1)
                    abc = (a + b + c) / 3.0
                    lam1 = abc - x1_term * cos(phi/3.0)
                    lam2 = abc + x1_term * cos((phi - M_PI)/3.0)
                    lam3 = abc + x1_term * cos((phi + M_PI)/3.0)
                    if abs_sort:
                        abs_lam1 = fabs(lam1)
                        abs_lam2 = fabs(lam2)
                        abs_lam3 = fabs(lam3)
                    else:
                        # reuse abs_lam variables without abs to avoid
                        # duplicate code below
                        abs_lam1 = lam1
                        abs_lam2 = lam2
                        abs_lam3 = lam3

                    if ascending:
                        if (abs_lam1 > abs_lam2):
                            stmp = lam1
                            lam1 = lam2
                            lam2 = stmp
                        if (abs_lam1 > abs_lam3):
                            stmp = lam3
                            lam3 = lam1
                            lam1 = stmp
                        if (abs_lam2 > abs_lam3):
                            stmp = lam3
                            lam3 = lam2
                            lam2 = stmp
                    else:
                        if (abs_lam3 > abs_lam2):
                            stmp = lam3
                            lam3 = lam2
                            lam2 = stmp
                        if (abs_lam3 > abs_lam1):
                            stmp = lam1
                            lam1 = lam3
                            lam3 = stmp
                        if (abs_lam2 > abs_lam1):
                            stmp = lam1
                            lam1 = lam2
                            lam2 = stmp
                    evals[0, pln, row, col] = lam1
                    evals[1, pln, row, col] = lam2
                    evals[2, pln, row, col] = lam3
    return np.asarray(evals)


def _real_symmetric_2x2_det(
    np_floats[:, ::1] M00, np_floats[:, ::1] M01, np_floats[:, ::1] M11
):
    """Determinant for real, symmetric 2 x 2 matrices."""
    cdef:
        cnp.float64_t m00, m01, m11;
        Py_ssize_t rows = M00.shape[0];
        Py_ssize_t cols = M00.shape[1];
        Py_ssize_t r, c;

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef np_floats[:, ::1] det = np.zeros((rows, cols), dtype=dtype)

    with nogil:
        for r in range(rows):
            for c in range(cols):
                m00 = <cnp.float64_t>(M00[r, c])
                m01 = <cnp.float64_t>(M01[r, c])
                m11 = <cnp.float64_t>(M11[r, c])
                det[r, c] = m00 * m11 - m01 * m01;
    return np.asarray(det)


def _real_symmetric_3x3_det(
    np_floats[:, :, ::1] M00, np_floats[:, :, ::1] M01, np_floats[:, :, ::1] M02,
    np_floats[:, :, ::1] M11, np_floats[:, :, ::1] M12, np_floats[:, :, ::1] M22,
):
    """Determinant for real, symmetric 3 x 3 matrices."""
    cdef:
        cnp.float64_t m00, m01, m02, m11, m12, m22;
        Py_ssize_t planes = M00.shape[0];
        Py_ssize_t rows = M00.shape[1];
        Py_ssize_t cols = M00.shape[2];
        Py_ssize_t pln, row, col;

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef np_floats[:, :, ::1] det = np.zeros((planes, rows, cols), dtype=dtype)

    with nogil:

        for pln in range(planes):
            for row in range(rows):
                for col in range(cols):
                    m00 = <cnp.float64_t>(M00[pln, row, col])
                    m01 = <cnp.float64_t>(M01[pln, row, col])
                    m02 = <cnp.float64_t>(M02[pln, row, col])
                    m11 = <cnp.float64_t>(M11[pln, row, col])
                    m12 = <cnp.float64_t>(M12[pln, row, col])
                    m22 = <cnp.float64_t>(M22[pln, row, col])
                    det[pln, row, col] = (
                        m00 * (m11 * m22 - m12 * m12)
                        - m01 * (m01 * m22 - m12 * m02)
                        + m02 * (m01 * m12 - m11 * m02)
                    )
    return np.asarray(det)


def _corner_moravec(np_floats[:, ::1] cimage, Py_ssize_t window_size=1):
    """Compute Moravec corner measure response image.

    This is one of the simplest corner detectors and is comparatively fast but
    has several limitations (e.g. not rotation invariant).

    Parameters
    ----------
    image : ndarray
        Input image.
    window_size : int, optional (default 1)
        Window size.

    Returns
    -------
    response : ndarray
        Moravec response image.

    References
    ----------
    .. [1] http://kiwi.cs.dal.ca/~dparks/CornerDetection/moravec.htm
    .. [2] https://en.wikipedia.org/wiki/Corner_detection

    Examples
    --------
    >>> from skimage.feature import corner_moravec
    >>> square = np.zeros([7, 7])
    >>> square[3, 3] = 1
    >>> square.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> corner_moravec(square).astype(int)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 2, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    """

    cdef Py_ssize_t rows = cimage.shape[0]
    cdef Py_ssize_t cols = cimage.shape[1]

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef np_floats[:, ::1] out = np.zeros((rows, cols), dtype=dtype)

    cdef np_floats msum, min_msum, t
    cdef Py_ssize_t r, c, br, bc, mr, mc, a, b

    with nogil:
        for r in range(2 * window_size, rows - 2 * window_size):
            for c in range(2 * window_size, cols - 2 * window_size):
                min_msum = DBL_MAX
                for br in range(r - window_size, r + window_size + 1):
                    for bc in range(c - window_size, c + window_size + 1):
                        if br != r and bc != c:
                            msum = 0
                            for mr in range(- window_size, window_size + 1):
                                for mc in range(- window_size, window_size + 1):
                                    t = cimage[r + mr, c + mc] - cimage[br + mr, bc + mc]
                                    msum += t * t
                            min_msum = min(msum, min_msum)

                out[r, c] = min_msum

    return np.asarray(out)


cdef inline np_floats _corner_fast_response(np_floats curr_pixel,
                                            np_floats* circle_intensities,
                                            signed char* bins, signed char
                                            state, char n) nogil:
    cdef char consecutive_count = 0
    cdef np_floats curr_response
    cdef Py_ssize_t l, m
    for l in range(15 + n):
        if bins[l % 16] == state:
            consecutive_count += 1
            if consecutive_count == n:
                curr_response = 0
                for m in range(16):
                    curr_response += fabs(circle_intensities[m] - curr_pixel)
                return curr_response
        else:
            consecutive_count = 0
    return 0


def _corner_fast(np_floats[:, ::1] image, signed char n, np_floats threshold):

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]

    cdef Py_ssize_t i, j, k

    cdef signed char speed_sum_b, speed_sum_d
    cdef np_floats curr_pixel
    cdef np_floats lower_threshold, upper_threshold
    cdef np_floats[:, ::1] corner_response = np.zeros((rows, cols),
                                                      dtype=dtype)

    cdef signed char *rp = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3,
                            -3, -2, -1]
    cdef signed char *cp = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1,
                            0, 1, 2, 3]
    cdef signed char bins[16]
    cdef np_floats circle_intensities[16]

    cdef cnp.float64_t curr_response

    with nogil:
        for i in range(3, rows - 3):
            for j in range(3, cols - 3):

                curr_pixel = image[i, j]
                lower_threshold = curr_pixel - threshold
                upper_threshold = curr_pixel + threshold

                for k in range(16):
                    circle_intensities[k] = image[i + rp[k], j + cp[k]]
                    if circle_intensities[k] > upper_threshold:
                        # Brighter pixel
                        bins[k] = b'b'
                    elif circle_intensities[k] < lower_threshold:
                        # Darker pixel
                        bins[k] = b'd'
                    else:
                        # Similar pixel
                        bins[k] = b's'

                # High speed test for n >= 12
                if n >= 12:
                    speed_sum_b = 0
                    speed_sum_d = 0
                    for k in range(0, 16, 4):
                        if bins[k] == b'b':
                            speed_sum_b += 1
                        elif bins[k] == b'd':
                            speed_sum_d += 1
                    if speed_sum_d < 3 and speed_sum_b < 3:
                        continue

                # Test for bright pixels
                curr_response = _corner_fast_response[np_floats](curr_pixel,
                                                      circle_intensities, bins,
                                                      b'b', n)

                # Test for dark pixels
                if curr_response == 0:
                    curr_response = _corner_fast_response[np_floats](curr_pixel,
                                                          circle_intensities,
                                                          bins, b'd', n)

                corner_response[i, j] = curr_response

    return np.asarray(corner_response)


def _corner_orientations(np_floats[:, ::1] image, Py_ssize_t[:, :] corners,
                         mask):
    """Compute the orientation of corners.

    The orientation of corners is computed using the first order central moment
    i.e. the center of mass approach. The corner orientation is the angle of
    the vector from the corner coordinate to the intensity centroid in the
    local neighborhood around the corner calculated using first order central
    moment.

    Parameters
    ----------
    image : 2D array
        Input grayscale image.
    corners : (N, 2) array
        Corner coordinates as ``(row, col)``.
    mask : 2D array
        Mask defining the local neighborhood of the corner used for the
        calculation of the central moment.

    Returns
    -------
    orientations : (N, 1) array
        Orientations of corners in the range [-pi, pi].

    References
    ----------
    .. [1] Ethan Rublee, Vincent Rabaud, Kurt Konolige and Gary Bradski
          "ORB : An efficient alternative to SIFT and SURF"
          http://www.vision.cs.chubu.ac.jp/CV-R/pdf/Rublee_iccv2011.pdf
    .. [2] Paul L. Rosin, "Measuring Corner Properties"
          http://users.cs.cf.ac.uk/Paul.Rosin/corner2.pdf

    Examples
    --------
    >>> from skimage.morphology import octagon
    >>> from skimage.feature import (corner_fast, corner_peaks,
    ...                              corner_orientations)
    >>> square = np.zeros((12, 12))
    >>> square[3:9, 3:9] = 1
    >>> square.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> corners = corner_peaks(corner_fast(square, 9), min_distance=1)
    >>> corners
    array([[3, 3],
           [3, 8],
           [8, 3],
           [8, 8]])
    >>> orientations = corner_orientations(square, corners, octagon(3, 2))
    >>> np.rad2deg(orientations)
    array([  45.,  135.,  -45., -135.])

    """

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    if mask.shape[0] % 2 != 1 or mask.shape[1] % 2 != 1:
        raise ValueError("Size of mask must be uneven.")

    cdef unsigned char[:, ::1] cmask = np.ascontiguousarray(mask != 0,
                                                            dtype=np.uint8)

    cdef Py_ssize_t i, r, c, r0, c0
    cdef Py_ssize_t mrows = mask.shape[0]
    cdef Py_ssize_t mcols = mask.shape[1]
    cdef Py_ssize_t mrows2 = (mrows - 1) / 2
    cdef Py_ssize_t mcols2 = (mcols - 1) / 2
    cdef np_floats[:, :] cimage = np.pad(image, (mrows2, mcols2),
                                         mode='constant',
                                         constant_values=0)
    cdef np_floats[:] orientations = np.zeros(corners.shape[0], dtype=dtype)
    cdef np_floats curr_pixel, m01, m10, m01_tmp

    with nogil:
        for i in range(corners.shape[0]):
            r0 = corners[i, 0]
            c0 = corners[i, 1]

            m01 = 0
            m10 = 0

            for r in range(mrows):
                m01_tmp = 0
                for c in range(mcols):
                    if cmask[r, c]:
                        curr_pixel = cimage[r0 + r, c0 + c]
                        m10 += curr_pixel * (c - mcols2)
                        m01_tmp += curr_pixel
                m01 += m01_tmp * (r - mrows2)

            orientations[i] = atan2(m01, m10)

    return np.asarray(orientations)
