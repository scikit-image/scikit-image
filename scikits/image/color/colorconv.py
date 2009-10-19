#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
:author: Nicolas Pinto, 2009
:license: modified BSD
"""

from __future__ import division

__all__ = ["rgb2hsv"]
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

