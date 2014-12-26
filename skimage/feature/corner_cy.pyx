#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
from libc.float cimport DBL_MAX
from libc.math cimport atan2

from ..util import img_as_float, pad
from ..color import rgb2grey

from .util import _prepare_grayscale_input_2D


def corner_moravec(image, Py_ssize_t window_size=1):
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
    .. [2] http://en.wikipedia.org/wiki/Corner_detection

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

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]

    cdef double[:, ::1] cimage = np.ascontiguousarray(img_as_float(image))
    cdef double[:, ::1] out = np.zeros(image.shape, dtype=np.double)

    cdef double msum, min_msum
    cdef Py_ssize_t r, c, br, bc, mr, mc, a, b
    for r in range(2 * window_size, rows - 2 * window_size):
        for c in range(2 * window_size, cols - 2 * window_size):
            min_msum = DBL_MAX
            for br in range(r - window_size, r + window_size + 1):
                for bc in range(c - window_size, c + window_size + 1):
                    if br != r and bc != c:
                        msum = 0
                        for mr in range(- window_size, window_size + 1):
                            for mc in range(- window_size, window_size + 1):
                                msum += (cimage[r + mr, c + mc]
                                         - cimage[br + mr, bc + mc]) ** 2
                        min_msum = min(msum, min_msum)

            out[r, c] = min_msum

    return np.asarray(out)


cdef inline double _corner_fast_response(double curr_pixel,
                                         double* circle_intensities,
                                         signed char* bins, signed char state, char n):
    cdef char consecutive_count = 0
    cdef double curr_response
    cdef Py_ssize_t l, m
    for l in range(15 + n):
        if bins[l % 16] == state:
            consecutive_count += 1
            if consecutive_count == n:
                curr_response = 0
                for m in range(16):
                    curr_response += abs(circle_intensities[m] - curr_pixel)
                return curr_response
        else:
            consecutive_count = 0
    return 0


def _corner_fast(double[:, ::1] image, signed char n, double threshold):

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]

    cdef Py_ssize_t i, j, k

    cdef signed char speed_sum_b, speed_sum_d
    cdef double curr_pixel
    cdef double lower_threshold, upper_threshold
    cdef double[:, ::1] corner_response = np.zeros((rows, cols),
                                                   dtype=np.double)

    cdef signed char *rp = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1]
    cdef signed char *cp = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3]
    cdef signed char bins[16]
    cdef double circle_intensities[16]

    cdef double curr_response

    for i in range(3, rows - 3):
        for j in range(3, cols - 3):

            curr_pixel = image[i, j]
            lower_threshold = curr_pixel - threshold
            upper_threshold = curr_pixel + threshold

            for k in range(16):
                circle_intensities[k] = image[i + rp[k], j + cp[k]]
                if circle_intensities[k] > upper_threshold:
                    # Brighter pixel
                    bins[k] = 'b'
                elif circle_intensities[k] < lower_threshold:
                    # Darker pixel
                    bins[k] = 'd'
                else:
                    # Similar pixel
                    bins[k] = 's'

            # High speed test for n >= 12
            if n >= 12:
                speed_sum_b = 0
                speed_sum_d = 0
                for k in range(0, 16, 4):
                    if bins[k] == 'b':
                        speed_sum_b += 1
                    elif bins[k] == 'd':
                        speed_sum_d += 1
                if speed_sum_d < 3 and speed_sum_b < 3:
                    continue

            # Test for bright pixels
            curr_response = \
                _corner_fast_response(curr_pixel, circle_intensities,
                                      bins, 'b', n)

            # Test for dark pixels
            if curr_response == 0:
                curr_response = \
                    _corner_fast_response(curr_pixel, circle_intensities,
                                          bins, 'd', n)

            corner_response[i, j] = curr_response

    return np.asarray(corner_response)


def corner_orientations(image, Py_ssize_t[:, :] corners, mask):
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

    image = _prepare_grayscale_input_2D(image)

    if mask.shape[0] % 2 != 1 or mask.shape[1] % 2 != 1:
        raise ValueError("Size of mask must be uneven.")

    cdef unsigned char[:, ::1] cmask = np.ascontiguousarray(mask != 0,
                                                            dtype=np.uint8)

    cdef Py_ssize_t i, r, c, r0, c0
    cdef Py_ssize_t mrows = mask.shape[0]
    cdef Py_ssize_t mcols = mask.shape[1]
    cdef Py_ssize_t mrows2 = (mrows - 1) / 2
    cdef Py_ssize_t mcols2 = (mcols - 1) / 2
    cdef double[:, :] cimage = pad(image, (mrows2, mcols2), mode='constant',
                                   constant_values=0)
    cdef double[:] orientations = np.zeros(corners.shape[0], dtype=np.double)
    cdef double curr_pixel
    cdef double m01, m10, m01_tmp

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
