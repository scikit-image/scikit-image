from __future__ import division
import numpy as np

__all__ = ['img_as_float', 'img_as_int', 'img_as_uint', 'img_as_ubyte']

from .. import get_log
log = get_log('dtype_converter')

dtype_range = {np.uint8: (0, 255),
               np.uint16: (0, 65535),
               np.int8: (-128, 127),
               np.int16: (-32768, 32767),
               np.float32: (0, 1),
               np.float64: (0, 1)}

integer_types = (np.uint8, np.uint16, np.int8, np.int16)


def _convert(image, dtype, prec_loss):
    """
    Convert an image to the requested data-type.

    Warnings are issues in case of precision loss, or when
    negative values have to be scaled into the positive domain.

    Parameters
    ----------
    image : ndarray
        Input image.
    dtype : dtype
        Target data-type.
    prec_loss : tuple
        List of input data-types that, when converted to `dtype`,
        would lose precision.

    """
    image = np.asarray(image)
    dtype_in = image.dtype.type

    if dtype_in == dtype:
        return image

    if dtype_in in prec_loss:
        log.warn('Possible precision loss, converting from '
                 '%s to %s' % (np.dtype(dtype_in), np.dtype(dtype)))

    try:
        imin, imax = dtype_range[dtype_in]
        omin, omax = dtype_range[dtype]
    except KeyError:
        raise ValueError("Unsure how to convert %s to %s." % \
                         (np.dtype(dtype_in), np.dtype(dtype)))

    sign_loss = (np.sign(imin) == -1) and (np.sign(omin) != -1)

    if sign_loss:
        log.warn('Possible sign loss when converting '
                 'negative image of type %s to positive '
                 'image of type %s.' % (np.dtype(dtype_in), np.dtype(dtype)))

    # If input type is non-negative, or if
    # converting to a positive-only type, then we
    # there's no need to shift numbers to the negative side
    if sign_loss or np.sign(imin) != -1:
        shift = 0
        omin = 0
    else:
        shift = omin

    scale = (omax - omin) / (imax - imin)

    if dtype in integer_types:
        round_fn = np.round
    else:
        round_fn = lambda x: x

    # Do scaling/shifting calculations in floating point
    image = image.astype(np.float64)
    out = image - imin
    out *= scale
    out += shift
    out = round_fn(out).astype(dtype)

    return out


def img_as_float(image):
    """Convert an image to double-precision floating point format.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    out : ndarray of float64
        Output image.

    Notes
    -----
    The range of a floating point image is [0, 1].
    Negative input values will be shifted to the positive domain.

    """
    prec_loss = ()
    return _convert(image, np.float64, prec_loss)


def img_as_uint(image):
    """Convert an image to 16-bit unsigned integer format.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    out : ndarray of uint16
        Output image.

    Notes
    -----
    Negative input values will be shifted to the positive domain.

    """

    prec_loss = (np.float32, np.float64)
    return _convert(image, np.uint16, prec_loss)


def img_as_int(image):
    """Convert an image to 16-bit signed integer format.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    out : ndarray of uint16
        Output image.

    Notes
    -----
    If the input data-type is positive-only (e.g., uint8), then
    the output image will still only have positive values.

    """
    prec_loss = (np.float32, np.float64, np.uint16)
    return _convert(image, np.int16, prec_loss)


def img_as_ubyte(image):
    """Convert an image to 8-bit unsigned integer format.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    out : ndarray of ubyte (uint8)
        Output image.

    Notes
    -----
    If the input data-type is positive-only (e.g., uint16), then
    the output image will still only have positive values.

    """
    prec_loss = (np.float32, np.float64, np.uint16, np.int16, np.int8)
    return _convert(image, np.ubyte, prec_loss)
