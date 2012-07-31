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
           'rgb2rgbcie', 'rgbcie2rgb', 'rgb2grey', 'rgb2gray', 'gray2rgb',
           'xyz2lab', 'lab2xyz', 'lab2rgb', 'rgb2lab'
           ]

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
    >>> from skimage import data
    >>> lena = data.lena()
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


def _prepare_colorarray(arr):
    """Check the shape of the array and convert it to
    floating point representation.

    """
    arr = np.asanyarray(arr)

    if arr.ndim != 3 or arr.shape[2] != 3:
        msg = "the input array must be have a shape == (.,.,3))"
        raise ValueError(msg)

    return dtype.img_as_float(arr)


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
    >>> from skimage import color
    >>> from skimage import data
    >>> lena = data.lena()
    >>> lena_hsv = color.rgb2hsv(lena)
    """
    arr = _prepare_colorarray(rgb)
    out = np.empty_like(arr)

    # -- V channel
    out_v = arr.max(-1)

    # -- S channel
    delta = arr.ptp(-1)
    # Ignore warning for zero divided by zero
    old_settings = np.seterr(invalid='ignore')
    out_s = delta / out_v
    out_s[delta == 0.] = 0.

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
    out_h[delta == 0.] = 0.

    np.seterr(**old_settings)

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
    >>> from skimage import data
    >>> lena = data.lena()
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

# CIE LAB constants for Observer= 2A, Illuminant= D65
lab_ref_white = np.array([0.95047, 1., 1.08883])

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
    >>> from skimage import data
    >>> from skimage.color import rgb2xyz, xyz2rgb
    >>> lena = data.lena()
    >>> lena_xyz = rgb2xyz(lena)
    >>> lena_rgb = xyz2rgb(lena_xyz)
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
    >>> from skimage import data
    >>> lena = data.lena()
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
    >>> from skimage import data
    >>> from skimage.color import rgb2rgbcie
    >>> lena = data.lena()
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
    >>> from skimage import data
    >>> from skimage.color import rgb2rgbcie, rgbcie2rgb
    >>> lena = data.lena()
    >>> lena_rgbcie = rgb2rgbcie(lena)
    >>> lena_rgb = rgbcie2rgb(lena_rgbcie)
    """
    return _convert(rgb_from_rgbcie, rgbcie)


def rgb2grey(rgb):
    """Compute luminance of an RGB image.

    Parameters
    ----------
    rgb : array_like
        The image in RGB format, in a 3-D array of shape (.., .., 3),
        or in RGBA format with shape (.., .., 4).

    Returns
    -------
    out : ndarray
        The luminance image, a 2-D array.

    Raises
    ------
    ValueError
        If `rgb2grey` is not a 3-D array of shape (.., .., 3) or
        (.., .., 4).

    References
    ----------
    .. [1] http://www.poynton.com/PDFs/ColorFAQ.pdf

    Notes
    -----
    The weights used in this conversion are calibrated for contemporary
    CRT phosphors::

        Y = 0.2125 R + 0.7154 G + 0.0721 B

    If there is an alpha channel present, it is ignored.

    Examples
    --------
    >>> from skimage.color import rgb2grey
    >>> from skimage import data
    >>> lena = data.lena()
    >>> lena_grey = rgb2grey(lena)
    """
    if rgb.ndim == 2:
        return rgb

    return _convert(grey_from_rgb, rgb[:, :, :3])[..., 0]

rgb2gray = rgb2grey


def gray2rgb(image):
    """Create an RGB representation of a grey-level image.

    Parameters
    ----------
    image : array_like
        Input image of shape ``(M, N)``.

    Returns
    -------
    rgb : ndarray
        RGB image of shape ``(M, N, 3)``.

    Raises
    ------
    ValueError
        If the input is not 2-dimensional.

    """
    if image.ndim != 2:
        raise ValueError('Gray-level image should be two-dimensional.')

    M, N = image.shape
    return np.dstack((image, image, image))


def xyz2lab(xyz):
    """XYZ to CIE-LAB color space conversion.

    Parameters
    ----------
    xyz : array_like
        The image in XYZ format, in a 3-D array of shape (.., .., 3).

    Returns
    -------
    out : ndarray
        The image in CIE-LAB format, in a 3-D array of shape (.., .., 3).

    Raises
    ------
    ValueError
        If `xyz` is not a 3-D array of shape (.., .., 3).

    Notes
    -----
    Observer= 2A, Illuminant= D65
    CIE XYZ tristimulus values x_ref = 95.047, y_ref = 100., z_ref = 108.883

    References
    ----------
    .. [1] http://www.easyrgb.com/index.php?X=MATH&H=07#text7
    .. [2] http://en.wikipedia.org/wiki/Lab_color_space

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2xyz, xyz2lab
    >>> lena = data.lena()
    >>> lena_xyz = rgb2xyz(lena)
    >>> lena_lab = xyz2lab(lena_xyz)
    """
    arr = _prepare_colorarray(xyz)

    # scale by CIE XYZ tristimulus values of the reference white point
    arr = arr / lab_ref_white

    # Nonlinear distortion and linear transformation
    mask = arr > 0.008856
    arr[mask] = np.power(arr[mask], 1. / 3.)
    arr[~mask] = 7.787 * arr[~mask] + 16. / 116.

    x, y, z = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Vector scaling
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    return np.dstack([L, a, b])


def lab2xyz(lab):
    """CIE-LAB to XYZcolor space conversion.

    Parameters
    ----------
    lab : array_like
        The image in lab format, in a 3-D array of shape (.., .., 3).

    Returns
    -------
    out : ndarray
        The image in XYZ format, in a 3-D array of shape (.., .., 3).

    Raises
    ------
    ValueError
        If `lab` is not a 3-D array of shape (.., .., 3).

    Notes
    -----
    Observer= 2A, Illuminant= D65
    CIE XYZ tristimulus values x_ref = 95.047, y_ref = 100., z_ref = 108.883

    References
    ----------
    .. [1] http://www.easyrgb.com/index.php?X=MATH&H=07#text7
    .. [2] http://en.wikipedia.org/wiki/Lab_color_space

    """

    arr = _prepare_colorarray(lab).copy()

    L, a, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)

    out = np.dstack([x, y, z])

    mask = out > 0.2068966
    out[mask] = np.power(out[mask], 3.)
    out[~mask] = (out[~mask] - 16.0 / 116.) / 7.787

    # rescale Observer= 2 deg, Illuminant= D65
    out *= lab_ref_white
    return out


def rgb2lab(rgb):
    """RGB to lab color space conversion.

    Parameters
    ----------
    rgb : array_like
        The image in RGB format, in a 3-D array of shape (.., .., 3).

    Returns
    -------
    out : ndarray
        The image in Lab format, in a 3-D array of shape (.., .., 3).

    Raises
    ------
    ValueError
        If `rgb` is not a 3-D array of shape (.., .., 3).

    Notes
    -----
    This function uses rgb2xyz and xyz2lab.
    """
    return xyz2lab(rgb2xyz(rgb))


def lab2rgb(lab):
    """Lab to RGB color space conversion.

    Parameters
    ----------
    rgb : array_like
        The image in Lab format, in a 3-D array of shape (.., .., 3).

    Returns
    -------
    out : ndarray
        The image in RGB format, in a 3-D array of shape (.., .., 3).

    Raises
    ------
    ValueError
        If `lab` is not a 3-D array of shape (.., .., 3).

    Notes
    -----
    This function uses lab2xyz and xyz2rgb.
    """
    return xyz2rgb(lab2xyz(lab))
