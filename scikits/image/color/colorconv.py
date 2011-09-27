#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions for converting between color spaces.

The "central" color space in this module is RGB, more specifically the linear
sRGB color space using D65 as a white-point [1]_.  This represents a
standard monitor (w/o gamma correction). For a good FAQ on color spaces see
[2]_.

The API consists of functions to convert to and from RGB as defined above, as
well as a generic function to convert to and from any supported color space
(which is done through RGB in most cases).


Supported color spaces
----------------------
* RGB : Red Green Blue.
        Here the sRGB standard [1]_.
* HSV : Hue, Saturation, Value.
        Uniquely defined when related to sRGB [3]_.
* RGB CIE : Red Green Blue.
        The original RGB CIE standard from 1931 [4]_. Primary colors are 700 nm
        (red), 546.1 nm (blue) and 435.8 nm (green).
* XYZ CIE : XYZ
        Derived from the RGB CIE color space. Chosen such that
        ``x == y == z == 1/3`` at the whitepoint, and all color matching
        functions are greater than zero everywhere.

:author: Nicolas Pinto (rgb2hsv)
:author: Ralf Gommers (hsv2rgb)
:author: Travis Oliphant (XYZ and RGB CIE functions)

:license: modified BSD

References
----------
.. [1] Official specification of sRGB, IEC 61966-2-1:1999.
.. [2] http://www.poynton.com/ColorFAQ.html
.. [3] http://en.wikipedia.org/wiki/HSL_and_HSV
.. [4] http://en.wikipedia.org/wiki/CIE_1931_color_space
"""

from __future__ import division

__all__ = ['convert_colorspace', 'rgb2hsv', 'hsv2rgb', 'rgb2xyz', 'xyz2rgb',
           'rgb2rgbcie', 'rgbcie2rgb', 'rgb2grey', 'rgb2gray']

__docformat__ = "restructuredtext en"

import numpy as np
from scipy import linalg
from ..util import dtype


def convert_colorspace(arr, fromspace, tospace):
    """Convert an image array to a new color space.

    Parameters
    ----------
    arr : array_like
        The image to convert.
    fromspace : str
        The color space to convert from. Valid color space strings are
        ['RGB', 'HSV', 'RGB CIE', 'XYZ']. Value may also be specified as lower
        case.
    tospace : str
        The color space to convert to. Valid color space strings are
        ['RGB', 'HSV', 'RGB CIE', 'XYZ']. Value may also be specified as lower
        case.

    Returns
    -------
    newarr : ndarray
        The converted image.

    Notes
    -----
    Conversion occurs through the "central" RGB color space, i.e. conversion
    from XYZ to HSV is implemented as XYZ -> RGB -> HSV instead of directly.

    Examples
    --------
    >>> import os
    >>> from scikits.image import data_dir
    >>> from scikits.image.io import imread

    >>> lena = imread(os.path.join(data_dir, 'lena.png'))
    >>> lena_hsv = convert_colorspace(lena, 'RGB', 'HSV')
    """
    fromdict = {'RGB': lambda im: im, 'HSV': hsv2rgb, 'RGB CIE': rgbcie2rgb,
                'XYZ': xyz2rgb}
    todict = {'RGB': lambda im: im, 'HSV': rgb2hsv, 'RGB CIE': rgb2rgbcie,
              'XYZ': rgb2xyz}

    fromspace = fromspace.upper()
    tospace = tospace.upper()
    if not fromspace in fromdict.keys():
        raise ValueError('fromspace needs to be one of %s' % fromdict.keys())
    if not tospace in todict.keys():
        raise ValueError('tospace needs to be one of %s' % todict.keys())

    return todict[tospace](fromdict[fromspace](arr))


def _prepare_colorarray(arr, dtype=np.float32):
    """Check the shape of the array, and give it the requested type."""
    arr = np.asanyarray(arr)

    if arr.ndim != 3 or arr.shape[2] != 3:
        msg = "the input array must be have a shape == (.,.,3))"
        raise ValueError(msg)

    return arr.astype(dtype)


def rgb2hsv(rgb):
    """RGB to HSV color space conversion.

    Parameters
    ----------
    rgb : array_like
        The image in RGB format, in a 3-D array of shape (.., .., 3).

    Returns
    -------
    out : ndarray
        The image in HSV format, in a 3-D array of shape (.., .., 3).

    Raises
    ------
    ValueError
        If `rgb` is not a 3-D array of shape (.., .., 3).

    Notes
    -----
    The conversion assumes an input data range of [0, 1] for all
    color components.

    Conversion between RGB and HSV color spaces results in some loss of
    precision, due to integer arithmetic and rounding [1]_.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/HSL_and_HSV

    Examples
    --------
    >>> import os
    >>> from scikits.image import data_dir
    >>> from scikits.image.io import imread

    >>> lena = imread(os.path.join(data_dir, 'lena.png'))
    >>> lena_hsv = color.rgb2hsv(lena)
    """
    arr = _prepare_colorarray(rgb)
    out = np.empty_like(arr)

    # -- V channel
    out_v = arr.max(-1)

    # -- S channel
    delta = arr.ptp(-1)
    out_s = delta / out_v
    out_s[delta == 0] = 0

    # -- H channel
    # red is max
    idx = (arr[:, :, 0] == out_v)
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]

    # green is max
    idx = (arr[:, :, 1] == out_v)
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]

    # blue is max
    idx = (arr[:, :, 2] == out_v)
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]
    out_h = (out[:, :, 0] / 6.) % 1.

    # -- output
    out[:, :, 0] = out_h
    out[:, :, 1] = out_s
    out[:, :, 2] = out_v

    # remove NaN
    out[np.isnan(out)] = 0

    return out


def hsv2rgb(hsv):
    """HSV to RGB color space conversion.

    Parameters
    ----------
    hsv : array_like
        The image in HSV format, in a 3-D array of shape (.., .., 3).

    Returns
    -------
    out : ndarray
        The image in RGB format, in a 3-D array of shape (.., .., 3).

    Raises
    ------
    ValueError
        If `hsv` is not a 3-D array of shape (.., .., 3).

    Notes
    -----
    The conversion assumes an input data range of [0, 1] for all
    color components.

    Conversion between RGB and HSV color spaces results in some loss of
    precision, due to integer arithmetic and rounding [1]_.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/HSL_and_HSV

    Examples
    --------
    >>> import os
    >>> from scikits.image import data_dir
    >>> from scikits.image.io import imread

    >>> lena = imread(os.path.join(data_dir, 'lena.png'))
    >>> lena_hsv = rgb2hsv(lena)
    >>> lena_rgb = hsv2rgb(lena_hsv)
    """
    arr = _prepare_colorarray(hsv)

    hi = np.floor(arr[:, :, 0] * 6)
    f = arr[:, :, 0] * 6 - hi
    p = arr[:, :, 2] * (1 - arr[:, :, 1])
    q = arr[:, :, 2] * (1 - f * arr[:, :, 1])
    t = arr[:, :, 2] * (1 - (1 - f) * arr[:, :, 1])
    v = arr[:, :, 2]

    hi = np.dstack([hi, hi, hi]).astype(np.uint8) % 6
    out = np.choose(hi, [np.dstack((v, t, p)),
                         np.dstack((q, v, p)),
                         np.dstack((p, v, t)),
                         np.dstack((p, q, v)),
                         np.dstack((t, p, v)),
                         np.dstack((v, p, q))])

    return out


#---------------------------------------------------------------
# Primaries for the coordinate systems
#---------------------------------------------------------------
cie_primaries = np.array([700, 546.1, 435.8])
sb_primaries = np.array([1. / 155, 1. / 190, 1. / 225]) * 1e5

#---------------------------------------------------------------
# Matrices that define conversion between different color spaces
#---------------------------------------------------------------

# From sRGB specification
xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
                          [0.212671, 0.715160, 0.072169],
                          [0.019334, 0.119193, 0.950227]])

rgb_from_xyz = linalg.inv(xyz_from_rgb)

# From http://en.wikipedia.org/wiki/CIE_1931_color_space
# Note: Travis's code did not have the divide by 0.17697
xyz_from_rgbcie = np.array([[0.49, 0.31, 0.20],
                            [0.17697, 0.81240, 0.01063],
                            [0.00, 0.01, 0.99]]) / 0.17697

rgbcie_from_xyz = linalg.inv(xyz_from_rgbcie)

# construct matrices to and from rgb:
rgbcie_from_rgb = np.dot(rgbcie_from_xyz, xyz_from_rgb)
rgb_from_rgbcie = np.dot(rgb_from_xyz, xyz_from_rgbcie)


grey_from_rgb = np.array([[0.2125, 0.7154, 0.0721],
                          [0, 0, 0],
                          [0, 0, 0]])

#-------------------------------------------------------------
# The conversion functions that make use of the matrices above
#-------------------------------------------------------------


def _convert(matrix, arr):
    """Do the color space conversion.

    Parameters
    ----------
    matrix : array_like
        The 3x3 matrix to use.
    arr : array_like
        The input array.

    Returns
    -------
    out : ndarray, dtype=float
        The converted array.
    """
    arr = dtype.img_as_float(arr)
    arr = _prepare_colorarray(arr)
    arr = np.swapaxes(arr, 0, 2)
    oldshape = arr.shape
    arr = np.reshape(arr, (3, -1))
    out = np.dot(matrix, arr)
    out.shape = oldshape
    out = np.swapaxes(out, 2, 0)

    return np.ascontiguousarray(out)


def xyz2rgb(xyz):
    """XYZ to RGB color space conversion.

    Parameters
    ----------
    xyz : array_like
        The image in XYZ format, in a 3-D array of shape (.., .., 3).

    Returns
    -------
    out : ndarray
        The image in RGB format, in a 3-D array of shape (.., .., 3).

    Raises
    ------
    ValueError
        If `xyz` is not a 3-D array of shape (.., .., 3).

    Notes
    -----
    The CIE XYZ color space is derived from the CIE RGB color space. Note
    however that this function converts to sRGB.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/CIE_1931_color_space

    Examples
    --------
    >>> import os
    >>> from scikits.image import data_dir
    >>> from scikits.image.io import imread
    >>> from scikits.image.color import rgb2xyz, xyz2rgb

    >>> lena = imread(os.path.join(data_dir, 'lena.png'))
    >>> lena_xyz = rgb2xyz(lena)
    >>> lena_rgb = xyz2rgb(lena_hsv)
    """
    return _convert(rgb_from_xyz, xyz)


def rgb2xyz(rgb):
    """RGB to XYZ color space conversion.

    Parameters
    ----------
    rgb : array_like
        The image in RGB format, in a 3-D array of shape (.., .., 3).

    Returns
    -------
    out : ndarray
        The image in XYZ format, in a 3-D array of shape (.., .., 3).

    Raises
    ------
    ValueError
        If `rgb` is not a 3-D array of shape (.., .., 3).

    Notes
    -----
    The CIE XYZ color space is derived from the CIE RGB color space. Note
    however that this function converts from sRGB.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/CIE_1931_color_space

    Examples
    --------
    >>> import os
    >>> from scikits.image import data_dir
    >>> from scikits.image.io import imread

    >>> lena = imread(os.path.join(data_dir, 'lena.png'))
    >>> lena_xyz = rgb2xyz(lena)
    """
    return _convert(xyz_from_rgb, rgb)


def rgb2rgbcie(rgb):
    """RGB to RGB CIE color space conversion.

    Parameters
    ----------
    rgb : array_like
        The image in RGB format, in a 3-D array of shape (.., .., 3).

    Returns
    -------
    out : ndarray
        The image in RGB CIE format, in a 3-D array of shape (.., .., 3).

    Raises
    ------
    ValueError
        If `rgb` is not a 3-D array of shape (.., .., 3).

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/CIE_1931_color_space

    Examples
    --------
    >>> import os
    >>> from scikits.image import data_dir
    >>> from scikits.image.io import imread
    >>> from scikits.image.color import rgb2rgbcie

    >>> lena = imread(os.path.join(data_dir, 'lena.png'))
    >>> lena_rgbcie = rgb2rgbcie(lena)
    """
    return _convert(rgbcie_from_rgb, rgb)


def rgbcie2rgb(rgbcie):
    """RGB CIE to RGB color space conversion.

    Parameters
    ----------
    rgbcie : array_like
        The image in RGB CIE format, in a 3-D array of shape (.., .., 3).

    Returns
    -------
    out : ndarray
        The image in RGB format, in a 3-D array of shape (.., .., 3).

    Raises
    ------
    ValueError
        If `rgbcie` is not a 3-D array of shape (.., .., 3).

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/CIE_1931_color_space

    Examples
    --------
    >>> import os
    >>> from scikits.image import data_dir
    >>> from scikits.image.io import imread
    >>> from scikits.image.color import rgb2rgbcie, rgbcie2rgb

    >>> lena = imread(os.path.join(data_dir, 'lena.png'))
    >>> lena_rgbcie = rgb2rgbcie(lena)
    >>> lena_rgb = rgbcie2rgb(lena_hsv)
    """
    return _convert(rgb_from_rgbcie, rgbcie)


def rgb2grey(rgb):
    """Compute luminance of an RGB image.

    Parameters
    ----------
    rgb : array_like
        The image in RGB format, in a 3-D array of shape (.., .., 3).

    Returns
    -------
    out : ndarray
        The luminance image, a 2-D array.

    Raises
    ------
    ValueError
        If `rgb2grey` is not a 3-D array of shape (.., .., 3).

    References
    ----------
    .. [1] http://www.poynton.com/PDFs/ColorFAQ.pdf

    Notes
    -----
    The weights used in this conversion are calibrated for contemporary
    CRT phosphors::

        Y = 0.2125 R + 0.7154 G + 0.0721 B

    Examples
    --------
    >>> import os
    >>> from scikits.image import data_dir
    >>> from scikits.image.io import imread
    >>> from scikits.image.color import rgb2grey

    >>> lena = imread(os.path.join(data_dir, 'lena.png'))
    >>> lena_grey = rgb2grey(lena)
    """
    return _convert(grey_from_rgb, rgb)[..., 0]

rgb2gray = rgb2grey
