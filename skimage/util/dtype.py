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

_supported_types = (np.uint8, np.uint16, np.uint32,
                    np.int8, np.int16, np.int32,
                    np.float16, np.float32, np.float64)


def _convert(image, dtype):
    """
    Convert an image to the requested data-type.

    Warnings are issued in case of precision loss, or when
    negative values have to be scaled into the positive domain.
    Floating point values must be in the range [0.0, 1.0].
    Numbers are not shifted to the negative side when converting from
    floating point or unsigned integer types to signed integer types.

    Parameters
    ----------
    image : ndarray
        Input image.
    dtype : dtype
        Target data-type.

    """
    image = np.asarray(image)
    dtype = np.dtype(dtype).type
    dtype_in = image.dtype.type
    dtypeobj = np.dtype(dtype)
    dtypeobj_in = np.dtype(dtype_in)
    kind = dtypeobj.kind
    kind_in = dtypeobj_in.kind
    itemsize = dtypeobj.itemsize
    itemsize_in = dtypeobj_in.itemsize

    if dtype_in == dtype:
        return image

    if not (dtype_in in _supported_types and dtype in _supported_types):
        raise ValueError("can not convert %s to %s." % (dtypeobj_in, dtypeobj))

    def sign_loss():
        log.warn("Possible sign loss when converting negative image of type "
                 "%s to positive image of type %s." % (dtypeobj_in, dtypeobj))

    def prec_loss():
        log.warn("Possible precision loss when converting from "
                 "%s to %s" % (dtypeobj_in, dtypeobj))

    if kind_in == 'f':
        if kind == 'f':
            # floating point -> floating point
            if itemsize_in > itemsize:
                prec_loss()
            return dtype(image)
        if kind == 'u':
            # floating point -> unsigned integer
            prec_loss()
            tdt = next(dt for dt in (dtype_in, np.float32, np.float64)
                       if itemsize * 8 < np.finfo(dt).nmant)
            image = tdt(image)
            umax = np.iinfo(dtype).max
            image = image * (umax + 1)
            np.clip(image, 0, umax, out=image)
            return dtype(image)
        if kind == 'i':
            # floating point -> signed integer
            prec_loss()
            tdt = next(dt for dt in (dtype_in, np.float32, np.float64)
                       if itemsize * 8 < np.finfo(dt).nmant)
            image = tdt(image)
            image = image * (np.iinfo(dtype).max + 1)
            np.clip(image, 0, np.iinfo(dtype).max, out=image)
            return dtype(image)
    if kind_in == 'u':
        if kind == 'f':
            # unsigned integer -> floating point
            if itemsize_in * 8 > np.finfo(dtype).nmant:
                prec_loss()
            tdt = next(dt for dt in (dtype, np.float32, np.float64)
                       if itemsize_in * 8 < np.finfo(dt).nmant)
            image = tdt(image)
            image /= np.iinfo(dtype_in).max
            return dtype(image)
        if kind == 'u':
            # unsigned integer -> unsigned integer
            if itemsize_in > itemsize:
                prec_loss()
                image = image >> 8 * (itemsize_in - itemsize)
                return dtype(image)
            else:
                result = dtype(image)
                result <<= 8 * (itemsize - itemsize_in)
                if itemsize - itemsize_in == 3:
                    # uint8 -> uint32
                    image = dtype(image)
                    image *= 2**16 + 2**8 + 1
                result += image
                return result
        if kind == 'i':
            # unsigned integer -> signed integer
            if itemsize_in >= itemsize:
                prec_loss()
                image = image >> (8 * (itemsize_in - itemsize) + 1)
                return dtype(image)
            else:
                result = dtype(image)
                result <<= 8 * (itemsize - itemsize_in) - 1
                if itemsize - itemsize_in == 3:
                    # uint8 -> int32
                    image = dtype(image)
                    image *= 2**16 + 2**8 + 1
                result += image >> 1
                return dtype(result)
    if kind_in == 'i':
        if kind == 'f':
            # signed integer -> floating point
            if itemsize_in * 8 > np.finfo(dtype).nmant:
                prec_loss()
            sign_loss()
            tdt = next(dt for dt in (dtype, np.float32, np.float64)
                       if itemsize_in * 8 < np.finfo(dt).nmant)
            image = tdt(image)
            image -= np.iinfo(dtype_in).min
            image /= np.iinfo(dtype_in).max - np.iinfo(dtype_in).min
            return dtype(image)
        if kind == 'u':
            # signed integer -> unsigned integer
            sign_loss()
            tdt = next(dt for dt in (np.int16, np.int32, np.int64)
                       if itemsize_in < np.dtype(dt).itemsize)
            image = tdt(image)
            image -= np.iinfo(dtype_in).min
            if itemsize_in == itemsize:
                return dtype(image)
            if itemsize_in > itemsize:
                prec_loss()
                image >>= 8 * (itemsize_in - itemsize)
                return dtype(image)
            else:
                result = dtype(image)
                result <<= 8 * (itemsize - itemsize_in)
                if itemsize - itemsize_in == 3:
                    # int8 -> uint32
                    image = dtype(image)
                    image *= 2**16 + 2**8 + 1
                result += image
                return result
        if kind == 'i':
            # signed integer -> signed integer
            if itemsize_in > itemsize:
                prec_loss()
                image = image // 2**(8 * (itemsize_in - itemsize))
                return dtype(image)
            else:
                tdt = next(dt for dt in (dtype, np.int16, np.int32, np.int64)
                           if itemsize_in < np.dtype(dt).itemsize)
                image = tdt(image)
                image -= np.iinfo(dtype_in).min
                tdt = next(dt for dt in (np.int32, np.int64)
                           if image.dtype.itemsize < np.dtype(dt).itemsize)
                result = tdt(image)
                result *= 2**(8 * (itemsize - itemsize_in))
                if itemsize - itemsize_in == 3:
                    # int8 -> int32
                    image = dtype(image)
                    image *= 2**16 + 2**8 + 1
                result += image
                result += np.iinfo(dtype).min
                return dtype(result)


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
    return _convert(image, np.float64)


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
    return _convert(image, np.uint16)


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
    return _convert(image, np.int16)


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
    return _convert(image, np.uint8)
