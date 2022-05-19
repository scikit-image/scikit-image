#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp

from libc.math cimport fabs

from .._shared.fused_numerics cimport np_floats


cpdef _nonmaximum_suppression_bilinear(
    np_floats[:, ::1] isobel,
    np_floats[:, ::1] jsobel,
    np_floats[:, ::1] magnitude,
    cnp.uint8_t[:, ::1] eroded_mask,
    float low_threshold=0
):
    """Apply non-maximum suppression to the magnitude image.

    Also applies the low_threshold.

    Parameters
    ----------
    isobel : np.ndarray
        The gradient along axis 0.
    jsobel : np.ndarray
        The gradient along axis 1.
    magnitude : np.ndarray
        The gradient magnitude.
    eroded_mask : np.ndarray
        Mask of pixels to include.
    low_threshold : float, optional
        Omit values where `magnitude` is lower than this threshold.

    Returns
    -------
    out : np.ndarray
        The gradient magnitude after application of non-maximum suppression and
        thresholding.
    """
    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef:
        Py_ssize_t x, y
        Py_ssize_t rows = magnitude.shape[0]
        Py_ssize_t cols = magnitude.shape[1]
        np_floats[:, ::1] out = np.empty((rows, cols), dtype=dtype)
        np_floats abs_isobel, abs_jsobel, w, neigh1, neigh2, m
        bint is_up, is_down, is_left, is_right, c_minus, c_plus

    if low_threshold == 0:
        # increment by epsilon so `m >= low_threshold` is False anywhere m == 0
        low_threshold = 1e-14

    with nogil:
        for x in range(magnitude.shape[0]):
            for y in range(magnitude.shape[1]):
                out[x, y] = 0
                m = magnitude[x, y]
                if not (eroded_mask[x, y] and (m >= low_threshold)):
                    continue

                is_down = (isobel[x, y] <= 0)
                is_up = 1 - is_down

                is_left = (jsobel[x, y] <= 0)
                is_right = 1 - is_left

                if (is_up and is_right) or (is_down and is_left):
                    abs_isobel = fabs(isobel[x, y])
                    abs_jsobel = fabs(jsobel[x, y])
                    if abs_isobel > abs_jsobel:
                        w = abs_jsobel / abs_isobel
                        neigh1 = magnitude[x + 1, y]
                        neigh2 = magnitude[x + 1, y + 1]
                        c_plus = (neigh2 * w + neigh1 * (1.0 - w)) <= m
                        neigh1 = magnitude[x - 1, y]
                        neigh2 = magnitude[x - 1, y - 1]
                        c_minus = (neigh2 * w + neigh1 * (1.0 - w)) <= m
                        if c_plus and c_minus:
                            out[x, y] = m
                    elif abs_isobel <= abs_jsobel:
                        w = abs_isobel / abs_jsobel
                        neigh1 = magnitude[x, y + 1]
                        neigh2 = magnitude[x + 1, y + 1]
                        c_plus = (neigh2 * w + neigh1 * (1.0 - w)) <= m
                        neigh1 = magnitude[x, y - 1]
                        neigh2 = magnitude[x - 1, y - 1]
                        c_minus = (neigh2 * w + neigh1 * (1.0 - w)) <= m
                        if c_plus and c_minus:
                            out[x, y] = m
                elif (is_down and is_right) or (is_up and is_left):
                    abs_isobel = fabs(isobel[x, y])
                    abs_jsobel = fabs(jsobel[x, y])
                    if abs_isobel < abs_jsobel:
                        w = abs_isobel / abs_jsobel
                        neigh1 = magnitude[x, y + 1]
                        neigh2 = magnitude[x - 1, y + 1]
                        c_plus = (neigh2 * w + neigh1 * (1.0 - w)) <= m
                        neigh1 = magnitude[x, y - 1]
                        neigh2 = magnitude[x + 1, y - 1]
                        c_minus = (neigh2 * w + neigh1 * (1.0 - w)) <= m
                        if c_plus and c_minus:
                            out[x, y] = m
                    elif abs_isobel >= abs_jsobel:
                        w = abs_jsobel / abs_isobel
                        neigh1 = magnitude[x - 1, y]
                        neigh2 = magnitude[x - 1, y + 1]
                        c_plus = (neigh2 * w + neigh1 * (1.0 - w)) <= m
                        neigh1 = magnitude[x + 1, y]
                        neigh2 = magnitude[x + 1, y - 1]
                        c_minus = (neigh2 * w + neigh1 * (1.0 - w)) <= m
                        if c_plus and c_minus:
                            out[x, y] = m
    return np.asarray(out)
