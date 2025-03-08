import warnings
from warnings import warn

import numpy as np

from .._shared.utils import deprecate_func


__all__ = [
    'img_as_float32',
    'img_as_float64',
    'img_as_float',
    'img_as_int',
    'img_as_uint',
    'img_as_ubyte',
    'img_as_bool',
    'rescale_to_float32',
    'rescale_to_float64',
    'rescale_to_float',
    'rescale_to_int16',
    'rescale_to_uint16',
    'rescale_to_uint8',
    'rescale_to_bool',
    'dtype_limits',
]

# Some of these may or may not be aliases depending on architecture & platform
_integer_types = (
    np.int8,
    np.byte,
    np.int16,
    np.short,
    np.int32,
    np.int64,
    np.longlong,
    np.int_,
    np.intp,
    np.intc,
    int,
    np.uint8,
    np.ubyte,
    np.uint16,
    np.ushort,
    np.uint32,
    np.uint64,
    np.ulonglong,
    np.uint,
    np.uintp,
    np.uintc,
)
_integer_ranges = {t: (np.iinfo(t).min, np.iinfo(t).max) for t in _integer_types}
dtype_range = {
    bool: (False, True),
    np.bool_: (False, True),
    float: (-1, 1),
    np.float16: (-1, 1),
    np.float32: (-1, 1),
    np.float64: (-1, 1),
}

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    # np.bool8 is a deprecated alias of np.bool_
    if hasattr(np, 'bool8'):
        dtype_range[np.bool8] = (False, True)

dtype_range.update(_integer_ranges)

_supported_types = list(dtype_range.keys())


def dtype_limits(image, clip_negative=False):
    """Return intensity limits, i.e. (min, max) tuple, of the image's dtype.

    Parameters
    ----------
    image : ndarray
        Input image.
    clip_negative : bool, optional
        If True, clip the negative range (i.e. return 0 for min intensity)
        even if the image dtype allows negative values.

    Returns
    -------
    imin, imax : tuple
        Lower and upper intensity limits.
    """
    imin, imax = dtype_range[image.dtype.type]
    if clip_negative:
        imin = 0
    return imin, imax


def _dtype_itemsize(itemsize, *dtypes):
    """Return first of `dtypes` with itemsize greater than `itemsize`

    Parameters
    ----------
    itemsize: int
        The data type object element size.

    Other Parameters
    ----------------
    *dtypes:
        Any Object accepted by `np.dtype` to be converted to a data
        type object

    Returns
    -------
    dtype: data type object
        First of `dtypes` with itemsize greater than `itemsize`.

    """
    return next(dt for dt in dtypes if np.dtype(dt).itemsize >= itemsize)


def _dtype_bits(kind, bits, itemsize=1):
    """Return dtype of `kind` that can store a `bits` wide unsigned int

    Parameters:
    kind: str
        Data type kind.
    bits: int
        Desired number of bits.
    itemsize: int
        The data type object element size.

    Returns
    -------
    dtype: data type object
        Data type of `kind` that can store a `bits` wide unsigned int

    """

    s = next(
        i
        for i in (itemsize,) + (2, 4, 8)
        if bits < (i * 8) or (bits == (i * 8) and kind == 'u')
    )

    return np.dtype(kind + str(s))


def _scale(a, n, m, copy=True):
    """Scale an array of unsigned/positive integers from `n` to `m` bits.

    Numbers can be represented exactly only if `m` is a multiple of `n`.

    Parameters
    ----------
    a : ndarray
        Input image array.
    n : int
        Number of bits currently used to encode the values in `a`.
    m : int
        Desired number of bits to encode the values in `out`.
    copy : bool, optional
        If True, allocates and returns new array. Otherwise, modifies
        `a` in place.

    Returns
    -------
    out : array
        Output image array. Has the same kind as `a`.
    """
    kind = a.dtype.kind
    if n > m and a.max() < 2**m:
        mnew = int(np.ceil(m / 2) * 2)
        if mnew > m:
            dtype = f'int{mnew}'
        else:
            dtype = f'uint{mnew}'
        n = int(np.ceil(n / 2) * 2)
        warn(
            f'Downcasting {a.dtype} to {dtype} without scaling because max '
            f'value {a.max()} fits in {dtype}',
            stacklevel=3,
        )
        return a.astype(_dtype_bits(kind, m))
    elif n == m:
        return a.copy() if copy else a
    elif n > m:
        # downscale with precision loss
        if copy:
            b = np.empty(a.shape, _dtype_bits(kind, m))
            np.floor_divide(a, 2 ** (n - m), out=b, dtype=a.dtype, casting='unsafe')
            return b
        else:
            a //= 2 ** (n - m)
            return a
    elif m % n == 0:
        # exact upscale to a multiple of `n` bits
        if copy:
            b = np.empty(a.shape, _dtype_bits(kind, m))
            np.multiply(a, (2**m - 1) // (2**n - 1), out=b, dtype=b.dtype)
            return b
        else:
            a = a.astype(_dtype_bits(kind, m, a.dtype.itemsize), copy=False)
            a *= (2**m - 1) // (2**n - 1)
            return a
    else:
        # upscale to a multiple of `n` bits,
        # then downscale with precision loss
        o = (m // n + 1) * n
        if copy:
            b = np.empty(a.shape, _dtype_bits(kind, o))
            np.multiply(a, (2**o - 1) // (2**n - 1), out=b, dtype=b.dtype)
            b //= 2 ** (o - m)
            return b
        else:
            a = a.astype(_dtype_bits(kind, o, a.dtype.itemsize), copy=False)
            a *= (2**o - 1) // (2**n - 1)
            a //= 2 ** (o - m)
            return a


def _normalize_float_0_to_1(image):
    """Normalize a floating point array in interval [0, 1].

    Parameters
    ----------
    image : ndarray
        Input image with floating dtype.

    Returns
    -------
    out : ndarray
        Normalized image with the same dtype as `image`.

    Notes
    -----
    This function deals with edge cases in the following way:

    - In case `image` contains inf, inf is normalized to NaN since inf / inf is
      not defined. All other values x are normalized to x / inf  which is 0.

    - Uniform arrays, where all pixels have the same value, are normalized to
      all-zero.
    """
    if image.dtype.kind != "f":
        msg = f"expected floating point, got {image.dtype=!r}"
        raise ValueError(msg)

    out = image.copy()  # always return a copy

    with np.errstate(all="raise"):
        float_min = out.min()
        float_max = out.max()

        try:
            float_ptp = float_max - float_min
        except FloatingPointError:
            # range is bigger than max float, half range to fit
            out /= 2
            float_min /= 2
            float_max /= 2
            float_ptp = float_max - float_min

        try:
            out /= float_ptp
            out -= float_min / float_ptp

        except FloatingPointError:
            if float_ptp == 0:
                msg = "normalizing uniform array to 0"
                warnings.warn(msg, category=RuntimeWarning, stacklevel=4)
                out = np.zeros_like(out)
            elif np.isinf(float_ptp):
                msg = "encountered inf, normalizing inf to NaN and other values to 0"
                warnings.warn(msg, category=RuntimeWarning, stacklevel=4)
                out = np.zeros_like(out)
                out[np.isinf(image)] = np.nan
            else:
                raise

    return out


def _convert(image, dtype, force_copy=False, uniform=False, *, legacy_float_range=None):
    """
    Convert an image to the requested data-type.

    Warnings are issued in case of precision loss, or when negative values
    are clipped during conversion to unsigned integer types (sign loss).

    Floating point values are expected to be normalized and will be clipped
    to the range [0.0, 1.0] or [-1.0, 1.0] when converting to unsigned or
    signed integers respectively.

    Numbers are not shifted to the negative side when converting from
    unsigned to signed integer types. Negative values will be clipped when
    converting to unsigned integers.

    Parameters
    ----------
    image : ndarray
        Input image.
    dtype : dtype
        Target data-type.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    uniform : bool, optional
        Uniformly quantize the floating point range to the integer range.
        By default (uniform=False) floating point values are scaled and
        rounded to the nearest integers, which minimizes back and forth
        conversion errors.

    .. versionchanged:: 0.15
        ``_convert`` no longer warns about possible precision or sign
        information loss. See discussions on these warnings at:
        https://github.com/scikit-image/scikit-image/issues/2602
        https://github.com/scikit-image/scikit-image/issues/543#issuecomment-208202228
        https://github.com/scikit-image/scikit-image/pull/3575

    legacy_float_range : bool, optional
        By default (``False``), the contents of integer images are
        scaled to the range [0.0, 1.0] if the target `dtype` is floating point.
        However, if legacy float range is enabled, images with signed integers
        will be scaled to [-1.0, 1.0] instead.

        .. versionadded:: 0.26

    References
    ----------
    .. [1] DirectX data conversion rules.
           https://msdn.microsoft.com/en-us/library/windows/desktop/dd607323%28v=vs.85%29.aspx
    .. [2] Data Conversions. In "OpenGL ES 2.0 Specification v2.0.25",
           pp 7-8. Khronos Group, 2010.
    .. [3] Proper treatment of pixels as integers. A.W. Paeth.
           In "Graphics Gems I", pp 249-256. Morgan Kaufmann, 1990.
    .. [4] Dirty Pixels. J. Blinn. In "Jim Blinn's corner: Dirty Pixels",
           pp 47-57. Morgan Kaufmann, 1998.

    """
    image = np.asarray(image)
    dtypeobj_in = image.dtype
    if dtype is np.floating:
        dtypeobj_out = np.dtype('float64')
    else:
        dtypeobj_out = np.dtype(dtype)
    dtype_in = dtypeobj_in.type
    dtype_out = dtypeobj_out.type
    kind_in = dtypeobj_in.kind
    kind_out = dtypeobj_out.kind
    itemsize_in = dtypeobj_in.itemsize
    itemsize_out = dtypeobj_out.itemsize

    # Below, we do an `issubdtype` check.  Its purpose is to find out
    # whether we can get away without doing any image conversion.  This happens
    # when:
    #
    # - the output and input dtypes are the same or
    # - when the output is specified as a type, and the input dtype
    #   is a subclass of that type (e.g. `np.floating` will allow
    #   `float32` and `float64` arrays through)

    if np.issubdtype(dtype_in, dtype):
        if force_copy:
            image = image.copy()
        if legacy_float_range is False and dtypeobj_in.kind == "f":
            image = _normalize_float_0_to_1(image)
        return image

    if not (dtype_in in _supported_types and dtype_out in _supported_types):
        raise ValueError(f'Cannot convert from {dtypeobj_in} to ' f'{dtypeobj_out}.')

    if kind_in in 'ui':
        imin_in = np.iinfo(dtype_in).min
        imax_in = np.iinfo(dtype_in).max
    if kind_out in 'ui':
        imin_out = np.iinfo(dtype_out).min
        imax_out = np.iinfo(dtype_out).max

    # any -> binary
    if kind_out == 'b':
        return image > dtype_in(dtype_range[dtype_in][1] / 2)

    # binary -> any
    if kind_in == 'b':
        result = image.astype(dtype_out)
        if kind_out != 'f':
            result *= dtype_out(dtype_range[dtype_out][1])
        return result

    # float -> any
    if kind_in == 'f':
        if kind_out == 'f':
            # float -> float
            if legacy_float_range is False:
                image = _normalize_float_0_to_1(image)
            return image.astype(dtype_out)

        if np.min(image) < -1.0 or np.max(image) > 1.0:
            raise ValueError("Images of type float must be between -1 and 1.")
        # floating point -> integer
        # use float type that can represent output integer type
        computation_type = _dtype_itemsize(
            itemsize_out, dtype_in, np.float32, np.float64
        )

        if not uniform:
            if kind_out == 'u':
                image_out = np.multiply(image, imax_out, dtype=computation_type)
            else:
                image_out = np.multiply(
                    image, (imax_out - imin_out) / 2, dtype=computation_type
                )
                image_out -= 1.0 / 2.0
            np.rint(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)
        elif kind_out == 'u':
            image_out = np.multiply(image, imax_out + 1, dtype=computation_type)
            np.clip(image_out, 0, imax_out, out=image_out)
        else:
            image_out = np.multiply(
                image, (imax_out - imin_out + 1.0) / 2.0, dtype=computation_type
            )
            np.floor(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)
        return image_out.astype(dtype_out)

    # signed/unsigned int -> float
    if kind_out == 'f':
        # use float type that can exactly represent input integers
        computation_type = _dtype_itemsize(
            itemsize_in, dtype_out, np.float32, np.float64
        )

        if kind_in == 'u':
            # using np.divide or np.multiply doesn't copy the data
            # until the computation time
            image = np.multiply(image, 1.0 / imax_in, dtype=computation_type)
            # DirectX uses this conversion also for signed ints
            # if imin_in:
            #     np.maximum(image, -1.0, out=image)
        elif kind_in == 'i':
            if legacy_float_range is True:
                # From DirectX conversions:
                # The most negative value maps to -1.0f
                # Every other value is converted to a float (call it c)
                # and then result = c * (1.0f / (2⁽ⁿ⁻¹⁾-1)).
                image = np.multiply(image, 1.0 / imax_in, dtype=computation_type)
                np.maximum(image, -1.0, out=image)
            elif legacy_float_range is False:
                ptp_in = imax_in - imin_in
                image = np.multiply(image, 1.0 / ptp_in, dtype=computation_type)
                image -= imin_in / ptp_in
            else:
                msg = (
                    "must set `legacy_float_range` to True or False "
                    "when rescaling from integers to float"
                )
                raise ValueError(msg)

        else:
            image = np.add(image, 0.5, dtype=computation_type)
            image *= 2 / (imax_in - imin_in)
            raise RuntimeError()

        return np.asarray(image, dtype_out)

    # unsigned int -> signed/unsigned int
    if kind_in == 'u':
        if kind_out == 'i':
            # unsigned int -> signed int
            image = _scale(image, 8 * itemsize_in, 8 * itemsize_out - 1)
            return image.view(dtype_out)
        else:
            # unsigned int -> unsigned int
            return _scale(image, 8 * itemsize_in, 8 * itemsize_out)

    # signed int -> unsigned int
    if kind_out == 'u':
        image = _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out)
        result = np.empty(image.shape, dtype_out)
        np.maximum(image, 0, out=result, dtype=image.dtype, casting='unsafe')
        return result

    # signed int -> signed int
    if itemsize_in > itemsize_out:
        return _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out - 1)

    image = image.astype(_dtype_bits('i', itemsize_out * 8))
    image -= imin_in
    image = _scale(image, 8 * itemsize_in, 8 * itemsize_out, copy=False)
    image += imin_out
    return image.astype(dtype_out)


def convert(image, dtype, force_copy=False, uniform=False):
    warn(
        "The use of this function is discouraged as its behavior may change "
        "dramatically in scikit-image 2.0. This function will be removed "
        "in scikit-image 2.0.",
        FutureWarning,
        stacklevel=2,
    )
    return _convert(image=image, dtype=dtype, force_copy=force_copy, uniform=uniform)


if _convert.__doc__ is not None:
    convert.__doc__ = (
        _convert.__doc__
        + """

    Warns
    -----
    FutureWarning:
        .. versionadded:: 0.17

        The use of this function is discouraged as its behavior may change
        dramatically in scikit-image 2.0. This function will be removed
        in scikit-image 2.0.
    """
    )


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0",
    hint="Use `skimage.util.rescale_to_float32(..., legacy_float_behavior=True)` "
    "instead.",
)
def img_as_float32(image, force_copy=False):
    """Convert an image to single-precision (32-bit) floating point format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of float32
        Output image.

    Notes
    -----
    The range of a floating point image is [0.0, 1.0] or [-1.0, 1.0] when
    converting from unsigned or signed datatypes, respectively.
    If the input image has a float type, intensity values are not modified
    and can be outside the ranges [0.0, 1.0] or [-1.0, 1.0].

    """
    return rescale_to_float32(image, force_copy=force_copy)


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0",
    hint="Use `skimage.util.rescale_to_float64(..., legacy_float_behavior=True)` "
    "instead.",
)
def img_as_float64(image, force_copy=False):
    """Convert an image to double-precision (64-bit) floating point format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of float64
        Output image.

    Notes
    -----
    The range of a floating point image is [0.0, 1.0] or [-1.0, 1.0] when
    converting from unsigned or signed datatypes, respectively.
    If the input image has a float type, intensity values are not modified
    and can be outside the ranges [0.0, 1.0] or [-1.0, 1.0].

    """
    return rescale_to_float64(image, force_copy=force_copy)


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0",
    hint="Use `skimage.util.rescale_to_float(..., legacy_float_behavior=True)` "
    "instead.",
)
def img_as_float(image, force_copy=False):
    """Convert an image to floating point format.

    This function is similar to :func:`~.rescale_to_float64`, but will not convert
    lower-precision floating point arrays to `float64`.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of float
        Output image.

    Notes
    -----
    The range of a floating point image is [0.0, 1.0] or [-1.0, 1.0] when
    converting from unsigned or signed datatypes, respectively.
    If the input image has a float type, intensity values are not modified
    and can be outside the ranges [0.0, 1.0] or [-1.0, 1.0].

    """
    return rescale_to_float(image, force_copy=force_copy)


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0",
    hint="Use `skimage.util.rescale_to_uint16` instead.",
)
def img_as_uint(image, force_copy=False):
    """Convert an image to 16-bit unsigned integer format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of uint16
        Output image.

    Notes
    -----
    Negative input values will be clipped.
    Positive values are scaled between 0 and 65535.

    """
    return rescale_to_uint16(image, force_copy=force_copy)


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0",
    hint="Use `skimage.util.rescale_to_int16` instead.",
)
def img_as_int(image, force_copy=False):
    """Convert an image to 16-bit signed integer format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of int16
        Output image.

    Notes
    -----
    The values are scaled between -32768 and 32767.
    If the input data-type is positive-only (e.g., uint8), then
    the output image will still only have positive values.

    """
    return rescale_to_int16(image, force_copy=force_copy)


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0",
    hint="Use `skimage.util.rescale_to_uint8` instead.",
)
def img_as_ubyte(image, force_copy=False):
    """Convert an image to 8-bit unsigned integer format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of ubyte (uint8)
        Output image.

    Notes
    -----
    Negative input values will be clipped.
    Positive values are scaled between 0 and 255.

    """
    return rescale_to_uint8(image, force_copy=force_copy)


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0",
    hint="Use `skimage.util.rescale_to_bool` instead.",
)
def img_as_bool(image, force_copy=False):
    """Convert an image to boolean format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of bool (`bool_`)
        Output image.

    Notes
    -----
    The upper half of the input dtype's positive range is True, and the lower
    half is False. All negative values (if present) are False.

    """
    return rescale_to_bool(image, force_copy=force_copy)


def rescale_to_float32(image, *, force_copy=False, legacy_float_range=False):
    """Convert an image to single-precision (32-bit) floating point format.

    As the name implies, this function will also rescale images. For integer
    images, it will map the minimal and maximal value supported by the
    respective integer to the value range [0.0, 1.0] (see `legacy_float_range`
    for previous legacy behavior). For floating images, it will map the minimal
    and maximal value to [0.0, 1.0].

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    legacy_float_range : bool, optional
        By default and if ``False``, the contents of integer images will be
        scaled to the range [0.0, 1.0]. However, if legacy behavior is enabled,
        images with signed integers will be scaled to [-1.0, 1.0] instead.
        This parameter as no effect on on-integer images.

    Returns
    -------
    out : ndarray of float32
        Output image.

    Notes
    -----
    When rescaling signed integers, the value 0 gets mapped close to 0.5. It
    is not exactly 0.5, since the negative value range of signed integers is
    one larger than the postive one, e.g., for int8 the range is [-128, 127].

    When scaling floating images containing inf, a warning is emitted. All
    values that are inf are assigned NaN, and all other values are assigned 0.

    When scaling uniform floating images, a warning is emitted and all values
    are assigned 0.

    Examples
    --------
    >>> import skimage as ski
    >>> import numpy as np

    >>> image_u8 = np.array([0, 255], dtype=np.uint8)
    >>> out = ski.util.rescale_to_float32(image_u8)
    >>> out
    array([0., 1.], dtype=float32)
    >>> out.dtype
    dtype('float32')

    >>> image_i8 = np.array([-128, 0, 127], dtype=np.int8)
    >>> ski.util.rescale_to_float32(image_i8)
    array([0.       , 0.5019608, 1.       ], dtype=float32)

    >>> ski.util.rescale_to_float32(image_i8, legacy_float_range=True)
    array([-1.,  0.,  1.], dtype=float32)
    """
    return _convert(
        image, np.float32, force_copy, legacy_float_range=legacy_float_range
    )


def rescale_to_float64(image, *, force_copy=False, legacy_float_range=False):
    """Convert an image to double-precision (64-bit) floating point format.

    As the name implies, this function will also rescale images. For integer
    images, it will map the minimal and maximal value supported by the
    respective integer to the value range [0.0, 1.0] (see `legacy_float_range`
    for previous legacy behavior). For floating images, it will map the minimal
    and maximal value to [0.0, 1.0].

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    legacy_float_range : bool, optional
        By default and if ``False``, the contents of integer images will be
        scaled to the range [0.0, 1.0]. However, if legacy behavior is enabled,
        images with signed integers will be scaled to [-1.0, 1.0] instead.
        This parameter as no effect on on-integer images.

    Returns
    -------
    out : ndarray of float64
        Output image.

    Notes
    -----
    When rescaling signed integers, the value 0 gets mapped close to 0.5. It
    is not exactly 0.5, since the negative value range of signed integers is
    one larger than the postive one, e.g., for int8 the range is [-128, 127].

    Examples
    --------
    >>> import skimage as ski
    >>> import numpy as np

    >>> image_u8 = np.array([0, 255], dtype=np.uint8)
    >>> out = ski.util.rescale_to_float64(image_u8)
    >>> out
    array([0., 1.])
    >>> out.dtype
    dtype('float64')

    >>> image_i8 = np.array([-128, 0, 127], dtype=np.int8)
    >>> ski.util.rescale_to_float64(image_i8)
    array([0.        , 0.50196078, 1.        ])

    >>> ski.util.rescale_to_float64(image_i8, legacy_float_range=True)
    array([-1.,  0.,  1.])
    """
    return _convert(
        image, np.float64, force_copy, legacy_float_range=legacy_float_range
    )


def rescale_to_float(image, *, force_copy=False, legacy_float_range=False):
    """Convert an image to floating point format.

    This function is similar to :func:`~.rescale_to_float64`, but will not
    convert lower-precision floating point arrays to `float64`.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    legacy_float_range : bool, optional
        By default and if ``False``, the contents of integer images will be
        scaled to the range [0.0, 1.0]. However, if legacy behavior is enabled,
        images with signed integers will be scaled to [-1.0, 1.0] instead.

    Returns
    -------
    out : ndarray of float
        Output image.

    Notes
    -----
    When rescaling signed integers, the value 0 gets mapped close to 0.5. It
    is not exactly 0.5, since the negative value range of signed integers is
    one larger than the postive one, e.g., for int8 the range is [-128, 127].

    Examples
    --------
    >>> import skimage as ski
    >>> import numpy as np

    >>> image_u8 = np.array([0, 255], dtype=np.uint8)
    >>> out = ski.util.rescale_to_float(image_u8)
    >>> out
    array([0., 1.])
    >>> out.dtype
    dtype('float64')

    >>> image_i8 = np.array([-128, 0, 127], dtype=np.int8)
    >>> ski.util.rescale_to_float(image_i8)
    array([0.        , 0.50196078, 1.        ])

    >>> ski.util.rescale_to_float(image_i8, legacy_float_range=True)
    array([-1.,  0.,  1.])
    """
    return _convert(
        image, np.floating, force_copy, legacy_float_range=legacy_float_range
    )


def rescale_to_uint16(image, *, force_copy=False):
    """Convert an image to 16-bit unsigned integer format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of uint16
        Output image.

    Notes
    -----
    Negative input values will be clipped.
    Positive values are scaled between 0 and 65535.

    """
    return _convert(image, np.uint16, force_copy)


def rescale_to_int16(image, *, force_copy=False):
    """Convert an image to 16-bit signed integer format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of int16
        Output image.

    Notes
    -----
    The values are scaled between -32768 and 32767.
    If the input data-type is positive-only (e.g., uint8), then
    the output image will still only have positive values.

    """
    return _convert(image, np.int16, force_copy)


def rescale_to_uint8(image, *, force_copy=False):
    """Convert an image to 8-bit unsigned integer format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of ubyte (uint8)
        Output image.

    Notes
    -----
    Negative input values will be clipped.
    Positive values are scaled between 0 and 255.

    """
    return _convert(image, np.uint8, force_copy)


def rescale_to_bool(image, *, force_copy=False):
    """Convert an image to boolean format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of bool (`bool_`)
        Output image.

    Notes
    -----
    The upper half of the input dtype's positive range is True, and the lower
    half is False. All negative values (if present) are False.

    """
    return _convert(image, bool, force_copy)
