# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as cnp


cdef inline Py_ssize_t _clip(Py_ssize_t x, Py_ssize_t low,
                             Py_ssize_t high) nogil:
    """Clips coordinate between high and low.

    This method was created so that `hessian_det_appx` does not have to make
    a Python call.

    Parameters
    ----------
    x : int
        Coordinate to be clipped.
    low : int
        The lower bound.
    high : int
        The higher bound.

    Returns
    -------
    x : int
        `x` clipped between `high` and `low`.
    """

    if(x > high):
        return high
    if(x < low):
        return low
    return x


cdef inline cnp.double_t _integ(
    cnp.double_t[:, ::1] img, Py_ssize_t r, Py_ssize_t c,
        Py_ssize_t rl, Py_ssize_t cl) nogil:
    """Integrate over the integral image in the given window

    This method was created so that `hessian_det_appx` does not have to make
    a Python call.

    Parameters
    ----------
    img : array
        The integral image over which to integrate.
    r : int
        The row number of the top left corner.
    c : int
        The column number of the top left corner.
    rl : int
        The number of rows over which to integrate.
    cl : int
        The number of columns over which to integrate.

    Returns
    -------
    ans : int
        The integral over the given window.
    """

    r = _clip(r, 0, img.shape[0] - 1)
    c = _clip(c, 0, img.shape[1] - 1)

    r2 = _clip(r + rl, 0, img.shape[0] - 1)
    c2 = _clip(c + cl, 0, img.shape[1] - 1)

    cdef cnp.double_t ans = img[r, c] + img[r2, c2] - img[r, c2] - img[r2, c]

    if (ans < 0):
        return 0
    return ans


def _hessian_matrix_det(cnp.double_t[:, ::1] img, double sigma):
    """Computes the approximate Hessian Determinant over an image.

    This method uses box filters over integral images to compute the
    approximate Hessian Determinant as described in [1]_.

    Parameters
    ----------
    img : array
        The integral image over which to compute Hessian Determinant.
    sigma : float
        Standard deviation used for the Gaussian kernel, used for the Hessian
        matrix

    Returns
    -------
    out : array
        The array of the Determinant of Hessians.

    References
    ----------
    .. [1] Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool,
           "SURF: Speeded Up Robust Features"
           ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf

    Notes
    -----
    The running time of this method only depends on size of the image. It is
    independent of `sigma` as one would expect. The downside is that the
    result for `sigma` less than `3` is not accurate, i.e., not similar to
    the result obtained if someone computed the Hessian and took it's
    determinant.
    """

    cdef Py_ssize_t size = int(3 * sigma)
    cdef Py_ssize_t height = img.shape[0]
    cdef Py_ssize_t width = img.shape[1]
    cdef Py_ssize_t r, c
    cdef Py_ssize_t s2 = (size - 1) / 2
    cdef Py_ssize_t s3 = size / 3
    cdef Py_ssize_t l = size / 3
    cdef Py_ssize_t w = size
    cdef Py_ssize_t b = (size - 1) / 2
    cdef cnp.double_t mid, side, tl, tr, bl, br
    cdef cnp.double_t[:, ::1] out = np.zeros_like(img, dtype=np.double)
    cdef cnp.double_t w_i = 1.0 / size / size

    cdef float dxx, dyy, dxy

    with nogil:
        if size % 2 == 0:
            size += 1

        for r in range(height):
            for c in range(width):
                tl = _integ(img, r - s3, c - s3, s3, s3)  # top left
                br = _integ(img, r + 1, c + 1, s3, s3)  # bottom right
                bl = _integ(img, r - s3, c + 1, s3, s3)  # bottom left
                tr = _integ(img, r + 1, c - s3, s3, s3)  # top right

                dxy = bl + tr - tl - br
                dxy = -dxy * w_i

                mid = _integ(img, r - s3 + 1, c - s2, 2 * s3 - 1, w)  # middle box
                side = _integ(img, r - s3 + 1, c - s3 / 2, 2 * s3 - 1, s3)  # sides

                dxx = mid - 3 * side
                dxx = -dxx * w_i

                mid = _integ(img, r - s2, c - s3 + 1, w, 2 * s3 - 1)
                side = _integ(img, r - s3 / 2, c - s3 + 1, s3, 2 * s3 - 1)

                dyy = mid - 3 * side
                dyy = -dyy * w_i

                out[r, c] = (dxx * dyy - 0.81 * (dxy * dxy))

    return out
