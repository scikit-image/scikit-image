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
        np_floats abs_isobel, abs_jsobel, w, m
        np_floats neigh1_1, neigh1_2, neigh2_1, neigh2_2
        bint is_up, is_down, is_left, is_right, c_minus, c_plus, cond1, cond2

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
                is_up = (isobel[x, y] >= 0)

                is_left = (jsobel[x, y] <= 0)
                is_right = (jsobel[x, y] >= 0)

                # Gradients are both positive or both negative
                cond1 = (is_up and is_right) or (is_down and is_left)
                # One gradient is negative while the other is positive
                cond2 = (is_down and is_right) or (is_up and is_left)
                if not cond1 and not cond2:
                    continue

                abs_isobel = fabs(isobel[x, y])
                abs_jsobel = fabs(jsobel[x, y])
                if cond1:
                    if abs_isobel > abs_jsobel:
                        w = abs_jsobel / abs_isobel
                        neigh1_1 = magnitude[x + 1, y]
                        neigh1_2 = magnitude[x + 1, y + 1]
                        neigh2_1 = magnitude[x - 1, y]
                        neigh2_2 = magnitude[x - 1, y - 1]
                    else:
                        w = abs_isobel / abs_jsobel
                        neigh1_1 = magnitude[x, y + 1]
                        neigh1_2 = magnitude[x + 1, y + 1]
                        neigh2_1 = magnitude[x, y - 1]
                        neigh2_2 = magnitude[x - 1, y - 1]
                elif cond2:
                    if abs_isobel < abs_jsobel:
                        w = abs_isobel / abs_jsobel
                        neigh1_1 = magnitude[x, y + 1]
                        neigh1_2 = magnitude[x - 1, y + 1]
                        neigh2_1 = magnitude[x, y - 1]
                        neigh2_2 = magnitude[x + 1, y - 1]
                    else:
                        w = abs_jsobel / abs_isobel
                        neigh1_1 = magnitude[x - 1, y]
                        neigh1_2 = magnitude[x - 1, y + 1]
                        neigh2_1 = magnitude[x + 1, y]
                        neigh2_2 = magnitude[x + 1, y - 1]
                # linear interpolation
                c_plus = (neigh1_2 * w + neigh1_1 * (1.0 - w)) <= m
                if c_plus:
                    c_minus = (neigh2_2 * w + neigh2_1 * (1.0 - w)) <= m
                    if c_minus:
                        out[x, y] = m
    return out.base
