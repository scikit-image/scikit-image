from __future__ import division
import numpy as np

__all__ = ['img_as_float', 'img_as_int', 'img_as_uint', 'img_as_ubyte']

from .. import get_log
log = get_log('dtype_converter')

dtype_range = {np.uint8: (0, 255),
               np.uint16: (0, 65535),
               np.int8: (-128, 127),
               np.int16: (-32768, 32767),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}

integer_types = (np.uint8, np.uint16, np.int8, np.int16)

_supported_types = (np.uint8, np.uint16, np.uint32,
                    np.int8, np.int16, np.int32,
                    np.float32, np.float64)

if np.__version__ >= "1.6.0":
    dtype_range[np.float16] = (-1, 1)
    _supported_types += (np.float16, )


def convert(image, dtype, force_copy=False, uniform=False):
    """
    Convert an image to the requested data-type.

    Warnings are issued in case of precision loss, or when
    negative values have to be scaled into the positive domain.

    Floating point values are expected to be normalized. They will be
    clipped to the range [0.0, 1.0] or [-1.0, 1.0] when converting to
    unsigned or signed integers respectively.

    Numbers are not shifted to the negative side when converting from
    unsigned to signed integer types. Negative values will be clipped from
    signed integers when converting to unsigned integers.

    Parameters
    ----------
    image : ndarray
        Input image.
    dtype : dtype
        Target data-type.
    force_copy : bool
        Force a copy of the data, irrespective of its current dtype.
    uniform : bool
        Quantize the floating point range to integer range uniformly instead
        of scaling and rounding floating point values to the nearest integers.

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
        if force_copy:
            image = image.copy()
        return image

    if not (dtype_in in _supported_types and dtype in _supported_types):
        raise ValueError("can not convert %s to %s." % (dtypeobj_in, dtypeobj))

    def sign_loss():
        log.warn("Possible sign loss when converting negative image of type "
                 "%s to positive image of type %s." % (dtypeobj_in, dtypeobj))

    def prec_loss():
        log.warn("Possible precision loss when converting from "
                 "%s to %s" % (dtypeobj_in, dtypeobj))

    def _dtype(itemsize, *dtypes):
        # Return first of `dtypes` with itemsize greater than `itemsize`
        return next(dt for dt in dtypes if itemsize < np.dtype(dt).itemsize)

    def _dtype2(kind, bits, itemsize=1):
        # Return dtype of `kind` that can store a `bits` wide unsigned int
        c = lambda x, y: x <= y if kind == 'u' else x < y
        s = next(i for i in (itemsize, ) + (2, 4, 8) if c(bits, i*8))
        return np.dtype(kind + str(s))

    def _scale(a, n, m, copy=True):
        # Scale unsigned integers from n to m bits
        # Numbers can be represented exactly only if m is a multiple of n
        # Output array is of same kind as input.
        kind = a.dtype.kind
        if n == m:
            return a.copy() if copy else a
        elif n > m:
            # downscale with precision loss
            prec_loss()
            if copy:
                b = np.empty(a.shape, _dtype2(kind, m))
                np.divide(a, 2**(n - m), out=b)
                return b
            else:
                a //= 2**(n - m)
                return a
        elif m % n == 0:
            # exact upscale to a multiple of n bits
            if copy:
                b = np.empty(a.shape, _dtype2(kind, m))
                np.multiply(a, (2**m - 1) / (2**n - 1), out=b)
                return b
            else:
                a = np.array(a, _dtype2(kind, m, a.dtype.itemsize), copy=False)
                a *= (2**m - 1) / (2**n - 1)
                return a
        else:
            # upscale to a multiple of n bits,
            # then downscale with precision loss
            prec_loss()
            o = (m // n + 1) * n
            if copy:
                b = np.empty(a.shape, _dtype2(kind, o))
                np.multiply(a, (2**o - 1) / (2**n - 1), out=b)
                b //= 2**(o - m)
                return b
            else:
                a = np.array(a, _dtype2(kind, o, a.dtype.itemsize), copy=False)
                a *= (2**o - 1) / (2**n - 1)
                a //= 2**(o - m)
                return a

    if kind_in == 'f':
        if kind == 'f':
            # floating point -> floating point
            if itemsize_in > itemsize:
                prec_loss()
            return dtype(image)

        # floating point -> integer
        prec_loss()
        # use float type that can represent output integer type
        image = np.array(image, _dtype(itemsize, dtype_in,
                                       np.float32, np.float64))
        if not uniform:
            if kind == 'u':
                image *= np.iinfo(dtype).max
            else:
                image *= np.iinfo(dtype).max - np.iinfo(dtype).min
                image -= 1.0
                image /= 2.0
            np.rint(image, out=image)
            np.clip(image, np.iinfo(dtype).min, np.iinfo(dtype).max, out=image)
        elif kind == 'u':
            image *= np.iinfo(dtype).max + 1
            np.clip(image, 0, np.iinfo(dtype).max, out=image)
        else:
            image += 1.0
            image *= (np.iinfo(dtype).max - np.iinfo(dtype).min + 1.0) / 2.0
            np.clip(image, np.iinfo(dtype).min, np.iinfo(dtype).max, out=image)
            image -= np.iinfo(dtype).min
        return dtype(image)

    if kind == 'f':
        # integer -> floating point
        if itemsize_in >= itemsize:
            prec_loss()
        # use float type that can exactly represent input integers
        image = np.array(image, _dtype(itemsize_in, dtype,
                                       np.float32, np.float64))
        if kind_in == 'u':
            image /= np.iinfo(dtype_in).max
            # DirectX uses this conversion also for signed ints
            #if np.iinfo(dtype_in).min:
            #    np.maximum(image, -1.0, out=image)
        else:
            image *= 2.0
            image += 1.0
            image /= np.iinfo(dtype_in).max - np.iinfo(dtype_in).min
        return dtype(image)

    if kind_in == 'u':
        if kind == 'i':
           # unsigned integer -> signed integer
            image = _scale(image, 8*itemsize_in, 8*itemsize-1)
            return image.view(dtype)
        else:
            # unsigned integer -> unsigned integer
            return _scale(image, 8*itemsize_in, 8*itemsize)

    if kind == 'u':
        # signed integer -> unsigned integer
        sign_loss()
        image = _scale(image, 8*itemsize_in-1, 8*itemsize)
        result = np.empty(image.shape, dtype)
        np.maximum(image, 0, out=result)
        return result

    # signed integer -> signed integer
    if itemsize_in > itemsize:
        return _scale(image, 8*itemsize_in-1, 8*itemsize-1)
    image = image.astype(_dtype2('i', itemsize*8))
    image -= np.iinfo(dtype_in).min
    image = _scale(image, 8*itemsize_in, 8*itemsize, copy=False)
    image += np.iinfo(dtype).min
    return image


def img_as_float(image, force_copy=False):
    """Convert an image to double-precision floating point format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of float64
        Output image.

    Notes
    -----
    The range of a floating point image is [0, 1].
    Negative input values will be shifted to the positive domain.

    """
    return convert(image, np.float64, force_copy)


def img_as_uint(image, force_copy=False):
    """Convert an image to 16-bit unsigned integer format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of uint16
        Output image.

    Notes
    -----
    Negative input values will be shifted to the positive domain.

    """
    return convert(image, np.uint16, force_copy)


def img_as_int(image, force_copy=False):
    """Convert an image to 16-bit signed integer format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of uint16
        Output image.

    Notes
    -----
    If the input data-type is positive-only (e.g., uint8), then
    the output image will still only have positive values.

    """
    return convert(image, np.int16, force_copy)


def img_as_ubyte(image, force_copy=False):
    """Convert an image to 8-bit unsigned integer format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of ubyte (uint8)
        Output image.

    Notes
    -----
    If the input data-type is positive-only (e.g., uint16), then
    the output image will still only have positive values.

    """
    return convert(image, np.uint8, force_copy)
