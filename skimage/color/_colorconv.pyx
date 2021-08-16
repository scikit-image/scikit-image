#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import cython
import numpy as np
from libc.math cimport floor

from .._shared.fused_numerics cimport np_floats


def rgb2hsv_inner(np_floats[:, ::1] rgb):
    """RGB to HSV color space conversion

    The original publication on HSV is [1]_. The form used here is from [2]_.

    References
    ----------
    .. [1] Smith, A. R. "Color gamut transform pairs", ACM SIGGRAPH Computer
           Graphics, 1978, 12(3), pp. 12–19.
           :DOI:`10.1145/965139.807361`
    .. [2] https://en.wikipedia.org/wiki/HSL_and_HSV#General_approach

    """
    cdef:
        np_floats[:, ::] hsv = np.empty_like(rgb)
        np_floats delta, minv, maxv, tmp
        Py_ssize_t i, n, ch

    n = rgb.shape[0]
    with nogil:
        for i in range(n):
            minv = maxv = rgb[i, 0]
            for ch in range(1, 3):
                tmp = rgb[i, ch]
                if tmp > maxv:
                    maxv = tmp
                elif tmp < minv:
                    minv = tmp
            delta = maxv - minv
            if delta == 0.0:
                hsv[i, :2] = 0.0
            else:
                hsv[i, 1] = delta / maxv
                if rgb[i, 0] == maxv:
                    hsv[i, 0] = (rgb[i, 1] - rgb[i, 2]) / delta
                    if hsv[i, 0] < 0:
                        hsv[i, 0] += 6.0
                elif rgb[i, 1] == maxv:
                    hsv[i, 0] = 2.0 + (rgb[i, 2] - rgb[i, 0]) / delta
                elif rgb[i, 2] == maxv:
                    hsv[i, 0] = 4.0 + (rgb[i, 0] - rgb[i, 1]) / delta
                hsv[i, 0] /= 6.0
            hsv[i, 2] = maxv
    return np.asarray(hsv)


def hsv2rgb_inner(np_floats[:, ::1] hsv):
    """HSV to RGB color space conversion

    The algorithm used here is given in [1]_.

    References
    ----------
    .. [1] Smith, A. R. "Color gamut transform pairs", ACM SIGGRAPH Computer
           Graphics, 1978, 12(3), pp. 12–19.
           :DOI:`10.1145/965139.807361`
    """
    cdef:
        np_floats[:, ::] rgb = np.empty_like(hsv)
        np_floats f, v, p
        Py_ssize_t i, n, hi, rem

    n = rgb.shape[0]
    with nogil:
        for i in range(n):
            hi = <int>floor(<double>(hsv[i, 0] * 6.0))
            f = hsv[i, 0] * 6 - hi
            v = hsv[i, 2]
            p = v * (1 - hsv[i, 1])
            rem = hi % 6
            if rem == 0:
                rgb[i, 0] = v
                rgb[i, 1] = v * (1 - (1 - f) * hsv[i, 1])
                rgb[i, 2] = p
            elif rem == 1:
                rgb[i, 0] = v * (1 - f * hsv[i, 1])
                rgb[i, 1] = v
                rgb[i, 2] = p
            elif rem == 2:
                rgb[i, 0] = p
                rgb[i, 1] = v
                rgb[i, 2] = v * (1 - (1 - f) * hsv[i, 1])
            elif rem == 3:
                rgb[i, 0] = p
                rgb[i, 1] = v * (1 - f * hsv[i, 1])
                rgb[i, 2] = v
            elif rem == 4:
                rgb[i, 0] = v * (1 - (1 - f) * hsv[i, 1])
                rgb[i, 1] = p
                rgb[i, 2] = v
            elif rem == 5:
                rgb[i, 0] = v
                rgb[i, 1] = p
                rgb[i, 2] = v * (1 - f * hsv[i, 1])
    return np.asarray(rgb)
