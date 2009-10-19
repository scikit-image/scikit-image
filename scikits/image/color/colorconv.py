#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
:author: Nicolas Pinto, 2009
:license: modified BSD
"""

from __future__ import division

__all__ = ['rgb2hsv', 'hsv2rgb']
__docformat__ = "restructuredtext en"

import numpy as np

def rgb2hsv(rgb):
    """RGB to HSV color space conversion.

    Parameters
    ----------
    rgb : ndarray
        The image in RGB format, in a 3-D array of shape (.., .., 3).

    Returns
    -------
    out : ndarray
        The image in HSV format, in a 3-D array of shape (.., .., 3).

    Raises
    ------
    ValueError
        If `rgb` is not a 3-D array of shape (.., .., 3).

    Examples
    --------
    >>> import os
    >>> from scikits.image import data_dir
    >>> from scikits.image.io import imread

    >>> lena = imread(os.path.join(data_dir, 'lena.png'))
    >>> lena_hsv = color.rgb2hsv(lena)
    """

    if type(rgb) != np.ndarray:
        raise TypeError, "the input array 'rgb' must be a numpy.ndarray"

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        msg = "the input array 'rgb' must be have a shape == (.,.,3))"
        raise ValueError, msg

    arr = rgb.astype("float32")
    out = np.empty_like(arr)

    # -- V channel
    out_v = arr.max(-1)

    # -- S channel
    delta = arr.ptp(-1)
    out_s = delta / out_v
    out_s[delta==0] = 0

    # -- H channel
    # red is max
    idx = (arr[:,:,0] == out_v)
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]

    # green is max
    idx = (arr[:,:,1] == out_v)
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0] ) / delta[idx]

    # blue is max
    idx = (arr[:,:,2] == out_v)
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1] ) / delta[idx]
    out_h = (out[:,:,0] / 6.) % 1.

    # -- output
    out[:,:,0] = out_h
    out[:,:,1] = out_s
    out[:,:,2] = out_v

    # remove NaN
    out[np.isnan(out)] = 0

    return out


def hsv2rgb(hsv):
    """HSV to RGB color space conversion.

    Parameters
    ----------
    hsv : ndarray
        The image in HSV format, in a 3-D array of shape (.., .., 3).

    Returns
    -------
    out : ndarray
        The image in RGB format, in a 3-D array of shape (.., .., 3).

    Raises
    ------
    ValueError
        If `hsv` is not a 3-D array of shape (.., .., 3).

    Examples
    --------
    >>> import os
    >>> from scikits.image import data_dir
    >>> from scikits.image.io import imread

    >>> lena = imread(os.path.join(data_dir, 'lena.png'))
    >>> lena_hsv = rgb2hsv(lena)
    >>> lena_rgb = hsv2rgb(lena_hsv)
    """

    if type(hsv) != np.ndarray:
        raise TypeError, "the input array 'hsv' must be a numpy.ndarray"

    if hsv.ndim != 3 or hsv.shape[2] != 3:
        msg = "the input array 'hsv' must be have a shape == (.,.,3))"
        raise ValueError, msg

    arr = hsv.astype("float32")

    hi = np.floor(arr[:,:,0] * 6)
    f = arr[:,:,0] * 6 - hi
    p = arr[:,:,2] * (1 - arr[:,:,1])
    q = arr[:,:,2] * (1 - f * arr[:,:,1])
    t = arr[:,:,2] * (1 - (1 - f) * arr[:,:,1])
    v = arr[:,:,2]

    hi = np.dstack([hi, hi, hi]).astype("uint8") % 6
    out = np.choose(hi, [np.dstack((v, t, p)),
                         np.dstack((q, v, p)),
                         np.dstack((p, v, t)),
                         np.dstack((p, q, v)),
                         np.dstack((t, p, v)),
                         np.dstack((v, p, q))])

    # remove NaN
    out[np.isnan(out)] = 0

    return out
