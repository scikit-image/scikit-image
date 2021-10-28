#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np


cdef np_real_numeric integrate(np_real_numeric[:, ::1] sat,
                               Py_ssize_t r0, Py_ssize_t c0,
                               Py_ssize_t r1, Py_ssize_t c1) nogil:
    """
    Using a summed area table / integral image, calculate the sum
    over a given window.

    This function is the same as the `integrate` function in
    `skimage.transform.integrate`, but this Cython version significantly
    speeds up the code.

    Parameters
    ----------
    sat : ndarray of np_real_numeric
        Summed area table / integral image.
    r0, c0 : int
        Top-left corner of block to be summed.
    r1, c1 : int
        Bottom-right corner of block to be summed.

    Returns
    -------
    S : np_real_numeric
        Sum over the given window.
    """
    cdef np_real_numeric S = 0

    S += sat[r1, c1]

    if (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S += sat[r0 - 1, c0 - 1]

    if (r0 - 1 >= 0):
        S -= sat[r0 - 1, c1]

    if (c0 - 1 >= 0):
        S -= sat[r1, c0 - 1]

    return S


def integral_image(image, *, dtype=None):
    r"""Integral image / summed area table.

    The integral image contains the sum of all elements above and to the
    left of it, i.e.:

    .. math::

       S[m, n] = \sum_{i \leq m} \sum_{j \leq n} X[i, j]

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    S : ndarray
        Integral image/summed area table of same shape as input image.

    Notes
    -----
    For better accuracy and to avoid potential overflow, the data type of the
    output may differ from the input's when the default dtype of None is used.
    For inputs with integer dtype, the behavior matches that for
    :func:`numpy.cumsum`. Floating point inputs will be promoted to at least
    double precision. The user can set `dtype` to override this behavior.

    References
    ----------
    .. [1] F.C. Crow, "Summed-area tables for texture mapping,"
           ACM SIGGRAPH Computer Graphics, vol. 18, 1984, pp. 207-212.

    """
    if dtype is None and image.real.dtype.kind == 'f':
        # default to at least double precision cumsum for accuracy
        dtype = np.promote_types(image.dtype, np.float64)

    S = image
    for i in range(image.ndim):
        S = S.cumsum(axis=i, dtype=dtype)
    return S
