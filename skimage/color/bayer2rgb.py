from scipy.ndimage import convolve, uniform_filter1d
import numpy as np

from ..util.dtype import convert


def _bayer2rgb_check_inputs(raw_image, bayer_pattern, dtype, out):
    if bayer_pattern not in ['rggb', 'grbg', 'bggr', 'gbrg']:
        raise ValueError('Unknown bayer_patter')

    if len(raw_image.shape) != 2:
        raise ValueError("Image must be a 2D image.")
    if raw_image.shape[0] % 2 != 0 or raw_image.shape[1] % 2 != 0:
        raise ValueError("Image must have an even number of rows and columns")

    if out is not None:
        # Many algorithms assume that the missing values start off as 0.
        # therefore, we enforce this assumption here.
        out.fill(0)
        dtype = out.dtype
    else:
        if dtype is None:
            dtype = raw_image.dtype
        out = np.zeros((raw_image.shape[0], raw_image.shape[1], 3),
                       dtype=dtype)

    dtype = np.dtype(dtype)

    try:
        from skimage.util.dtype import check_precision_loss
    except ImportError:
        def check_precision_loss(*args, **kwargs):
            pass
    check_precision_loss(raw_image.dtype, dtype,
                         output_warning=True,
                         int_same_size_lossy=True)

    return (raw_image, bayer_pattern, dtype, out)


def bayer2rgb(raw_image, bayer_pattern='rggb', dtype=None, *, out=None):
    """Convert a raw image from a sensor with bayer filter to a rgb image.

    The NxM image is converted in 3 steps:
        1. A NxMx3 canvas where the 3rd dimension corresponds to RGB.
        2. The values where red green or blue were measured directly
           are assigned from the RAW image.
        3. The missing values are in-filled using the appropriate algorithm.

    The algorithm implemented uses a convolution to in-fill the missing values.
    The green channel uses the following filter _[1]:

        K_green = 1 / 4 * np.array([[0, 1, 0],
                                    [1, 4, 1],
                                    [0, 1, 0]], dtype='float32')

    while the red and blue pixels use:

        K_red_or_blue = 1 / 4 * np.array([[1, 2, 1],
                                          [2, 4, 2],
                                          [1, 2, 1]], dtype='float32')

    Follows the discussion in
    [1] Color Image and Video Enhancement, Celebi, M.E, Leccca M: Smolka B,
    2015. Chapter 2: Image Demosaicing Ruiwen Zhen and Robert L. Stevenson

    Notes
    =====

    _[1] mentions that this technique introduces artifacts especially at sharp
    color boundaries. In fact they do not recommend the use of a 3x3
    convolution filter to colorize images.

    Parameters
    ==========
    raw_image: np.array [N, M]
        N and M must be multiples of 2. This contains the raw data as measured
        from the sensor.

    bayer_pattern: ['rggb',  'grbg', 'bggr', 'gbrg']
        The bayer pattern of the sensor flattened in C-order.

    dtype: np.dtype, optional
        Output image type. If the output image has more precision than the
        input image, then it is used during the computation. Otherwise, the
        computation is done using the input image type and the result is
        converted to the output image type.

        For integer input types, images are scaled down prior to convolution
        to avoid overflow errors.

    out: np.ndarray [N, M, 3], optional
        Output RGB image. If provided, the dtype parameter is ignored.

    Returns
    =======
    np.ndarray [N, M, 3]
        RGB image.

    """

    # The implementation has been unrolled to improve speed.
    # If anybody knows a fast, more readible implementation,
    # please change this unrolled one.
    # It works about
    # 50% as fast for float32
    # 10% faster for float64
    # 60% faster uint8
    # See _bayer2rgb_naive for unrolled implemementation

    (raw_image, bayer_pattern, dtype, color_image) = _bayer2rgb_check_inputs(
        raw_image, bayer_pattern, dtype, out)

    # These functions are defined so as to allow floating pointers to use
    # True divide, while allowing integer types to floor divide and then
    # add avoiding overflow errors
    if dtype.kind == 'f':
        def divide_by_2(array):
            return array * np.array(0.5, dtype=dtype)

        def add_divide_by_2(array1, array2):
            return (array1 + array2) * np.array(0.5, dtype=dtype)

        def add_divide_by_4(array1, array2):
            return (array1 + array2) * np.array(0.25, dtype=dtype)

    else:
        def divide_by_2(array):
            return array // 2

        def add_divide_by_2(array1, array2):
            return array1 // 2 + array2 // 2

        def add_divide_by_4(array1, array2):
            return add_divide_by_2(array1, array2) // 2

    # Create convenient views
    # These views have for their first two indicies the pixels "mega pixels"
    # that contain something like
    # rg
    # gb
    # The last two indicies are the index of the subpixel within it
    red_image = color_image[:, :, 0]
    red_image.shape = (red_image.shape[0] // 2, 2, red_image.shape[1] // 2, 2)
    red_image = np.swapaxes(red_image, 1, 2)

    green_image = color_image[:, :, 1]
    green_image.shape = (
        raw_image.shape[0] // 2, 2, raw_image.shape[1] // 2, 2)
    green_image = np.swapaxes(green_image, 1, 2)

    blue_image = color_image[:, :, 2]
    blue_image.shape = (raw_image.shape[0] // 2, 2, raw_image.shape[1] // 2, 2)
    blue_image = np.swapaxes(blue_image, 1, 2)

    # TODO: allow convert to take in the "output" image
    #       this helps for large arrays, but maybe for small arrays too
    # convert(raw_image[0::2, 0::2], output=red_image[:, :, 0, 0])
    # convert(raw_image[1::2, 1::2], output=blue_image[:, :, 1, 1])
    # convert(raw_image[0::2, 1::2], output=green_image[:, :, 0, 1])
    # convert(raw_image[1::2, 0::2], output=green_image[:, :, 1, 0])

    def infill_red_or_blue_00(rb):
        rb[:, :, 0, 0] = convert(raw_image[0::2, 0::2], dtype=dtype)
        # Compute this one first, because if the array is C continuous, this
        # Each line here is on the same cache line
        # Adjacent pixels
        rb[:, :-1, 0, 1] = add_divide_by_2(rb[:, :-1, 0, 0], rb[:, 1:, 0, 0])
        rb[:, -1, 0, 1] = rb[:, -1, 0, 0]

        # This actually takes care of the "corner" pixel because
        # The values around that one pixel have now been filled in
        rb[:-1, :, 1, :] = add_divide_by_2(rb[:-1, :, 0, :], rb[1:, :, 0, :])
        rb[-1, :, 1, :] = rb[-1, :, 0, :]

    def infill_red_or_blue_01(rb):
        rb[:, :, 0, 1] = convert(raw_image[0::2, 1::2], dtype=dtype)
        rb[:, 1:, 0, 0] = add_divide_by_2(rb[:, 1:, 0, 1], rb[:, :-1, 0, 1])
        rb[:, 0, 0, 0] = rb[:, 0, 0, 1]
        rb[:-1, :, 1, :] = add_divide_by_2(rb[:-1, :, 0, :], rb[1:, :, 0, :])
        rb[-1, :, 1, :] = rb[-1, :, 0, :]

    def infill_red_or_blue_10(rb):
        rb[:, :, 1, 0] = convert(raw_image[1::2, 0::2], dtype=dtype)
        rb[:, :-1, 1, 1] = add_divide_by_2(rb[:, :-1, 1, 0], rb[:, 1:, 1, 0])
        rb[:, -1, 1, 1] = rb[:, -1, 1, 0]
        rb[1:, :, 0, :] = add_divide_by_2(rb[1:, :, 1, :], rb[:-1, :, 1, :])
        rb[0, :, 0, :] = rb[0, :, 1, :]

    def infill_red_or_blue_11(rb):
        rb[:, :, 1, 1] = convert(raw_image[1::2, 1::2], dtype=dtype)
        rb[:, 1:, 1, 0] = add_divide_by_2(rb[:, :-1, 1, 1], rb[:, 1:, 1, 1])
        rb[:, 0, 1, 0] = rb[:, 0, 1, 1]
        rb[1:, :, 0, :] = add_divide_by_2(rb[:-1, :, 1, :], rb[1:, :, 1, :])
        rb[0, :, 0, :] = rb[0, :, 1, :]

    def infill_green_01(g):
        g[:, :, 1, 0] = convert(raw_image[1::2, 0::2], dtype=dtype)
        g[:, :, 0, 1] = convert(raw_image[0::2, 1::2], dtype=dtype)
        # Compute the convolution horizontally
        g[:, 1:, 0, 0] = add_divide_by_4(g[:, :-1, 0, 1], g[:, 1:, 0, 1])
        g[:, 0, 0, 0] = divide_by_2(g[:, 0, 0, 1])

        g[:, -1, 1, 1] = divide_by_2(g[:, -1, 1, 0])
        g[:, :-1, 1, 1] = add_divide_by_4(g[:, :-1, 1, 0], g[:, 1:, 1, 0])

        # Now compute it vertically
        g[1:, :, 0, 0] += add_divide_by_4(g[1:, :, 1, 0], g[:-1, :, 1, 0])
        g[0, :, 0, 0] += divide_by_2(g[0, :, 1, 0])

        g[:-1, :, 1, 1] += add_divide_by_4(g[:-1, :, 0, 1], g[1:, :, 0, 1])
        g[-1, :, 1, 1] += divide_by_2(g[-1, :, 0, 1])

    def infill_green_00(g):
        g[:, :, 0, 0] = convert(raw_image[0::2, 0::2], dtype=dtype)
        g[:, :, 1, 1] = convert(raw_image[1::2, 1::2], dtype=dtype)
        g[:, 1:, 1, 0] = add_divide_by_4(g[:, :-1, 1, 1], g[:, 1:, 1, 1])
        g[:, 0, 1, 0] = divide_by_2(g[:, 0, 1, 1])

        g[:, -1, 0, 1] = divide_by_2(g[:, -1, 0, 0])
        g[:, :-1, 0, 1] = add_divide_by_4(g[:, :-1, 0, 0], g[:, 1:, 0, 0])

        g[0, :, 0, 1] += divide_by_2(g[0, :, 1, 1])
        g[-1, :, 1, 0] += divide_by_2(g[-1, :, 0, 0])

        g[1:, :, 0, 1] += add_divide_by_4(g[1:, :, 1, 1], g[:-1, :, 1, 1])
        g[:-1, :, 1, 0] += add_divide_by_4(g[:-1, :, 0, 0], g[1:, :, 0, 0])

    if bayer_pattern[0] == 'r':
        infill_red_or_blue_00(red_image)
    elif bayer_pattern[1] == 'r':
        infill_red_or_blue_01(red_image)
    elif bayer_pattern[2] == 'r':
        infill_red_or_blue_10(red_image)
    elif bayer_pattern[3] == 'r':
        infill_red_or_blue_11(red_image)

    if bayer_pattern[0] == 'b':
        infill_red_or_blue_00(blue_image)
    elif bayer_pattern[1] == 'b':
        infill_red_or_blue_01(blue_image)
    elif bayer_pattern[2] == 'b':
        infill_red_or_blue_10(blue_image)
    elif bayer_pattern[3] == 'b':
        infill_red_or_blue_11(blue_image)

    if bayer_pattern[0] == 'g':
        infill_green_00(green_image)
    elif bayer_pattern[1] == 'g':
        infill_green_01(green_image)

    return color_image


def bayer2rgb_naive(raw_image, bayer_pattern='rggb', dtype=None, *, out=None):
    """
    This is a naive implementation using scipy's convolve.

    I think there are 2 main issues:
    1. Having the color channel as the last index in C makes each channel
       not contiguous. This likely a source of slowdown in the convolution
       implemention of scipy.
    2. The convolution doesn't take into account 0 coefficients. This makes
       it particularly require 4 more floating point operations per pixel.

    """
    (raw_image, bayer_pattern, dtype, out) = _bayer2rgb_check_inputs(
        raw_image, bayer_pattern, dtype, out)

    K_green = np.array([[0, 1, 0],
                        [1, 4, 1],
                        [0, 1, 0]], dtype=dtype)
    K_red_or_blue = np.array([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]], dtype=dtype)

    def prepare_red_or_blue_00(rb, raw_image):
        rb[::2, ::2] = convert(raw_image[0::2, 0::2], dtype=dtype)

    def prepare_green_00(g, raw_image):
        g[0::2, 0::2] = convert(raw_image[0::2, 0::2], dtype=dtype)
        g[1::2, 1::2] = convert(raw_image[1::2, 1::2], dtype=dtype)

    if 'r' == bayer_pattern[0]:
        prepare_red_or_blue_00(out[..., 0], raw_image)
    elif 'r' == bayer_pattern[1]:
        prepare_red_or_blue_00(out[:, ::-1, 0], raw_image[:, ::-1])
    elif 'r' == bayer_pattern[2]:
        prepare_red_or_blue_00(out[::-1, :, 0], raw_image[::-1, :])
    else:  # 'r' == bayer_patter[2]:
        prepare_red_or_blue_00(out[::-1, ::-1, 0], raw_image[::-1, ::-1])

    if 'g' == bayer_pattern[0]:
        prepare_green_00(out[..., 1], raw_image)
    else:  # 'g' == bayer_pattern[1]:
        prepare_green_00(out[::-1, :, 1], raw_image[::-1, :])

    if 'b' == bayer_pattern[0]:
        prepare_red_or_blue_00(out[..., 2], raw_image)
    elif 'b' == bayer_pattern[1]:
        prepare_red_or_blue_00(out[:, ::-1, 2], raw_image[:, ::-1])
    elif 'b' == bayer_pattern[2]:
        prepare_red_or_blue_00(out[::-1, :, 2], raw_image[::-1, :])
    else:  # 'b' == bayer_patter[3]:
        prepare_red_or_blue_00(out[::-1, ::-1, 2], raw_image[::-1, ::-1])

    if dtype.kind == 'f':
        # Predivide the small array
        K_green /= 4
        K_red_or_blue /= 4
    else:
        # Can't divide K, would get 0, so remove significant bits from out
        out //= 4

    # do not use the out parameter of convolve.
    # It assumes the input and out arrays are different.
    out[:, :, 0] = convolve(out[:, :, 0], K_red_or_blue, mode='mirror')
    out[:, :, 1] = convolve(out[:, :, 1], K_green, mode='mirror')
    out[:, :, 2] = convolve(out[:, :, 2], K_red_or_blue, mode='mirror')

    return out


def bayer2rgb_slicing(raw_image, bayer_pattern='rggb', dtype=None,
                      *, out=None):
    """
    This implementation uses symmetry and slicing to attempt to simplify
    the impementation of the in-filling.

    Unfortunately, strided operations aren't well supported by numpy
    so it seems that this incurs more memory copies that implementing
    each of the 4 cases by hand.
    """
    (raw_image, bayer_pattern, dtype, color_image) = _bayer2rgb_check_inputs(
        raw_image, bayer_pattern, dtype, out)

    # These functions are defined so as to allow floating pointers to use
    # True divide, while allowing integer types to floor divide and then
    # add avoiding overflow errors
    if dtype.kind == 'f':
        def divide_by_2(array):
            return array * np.array(0.5, dtype=dtype)

        def add_divide_by_2(array1, array2):
            return (array1 + array2) * np.array(0.5, dtype=dtype)

        def add_divide_by_4(array1, array2):
            return (array1 + array2) * np.array(0.25, dtype=dtype)

    else:
        def divide_by_2(array):
            return array // 2

        def add_divide_by_2(array1, array2):
            return array1 // 2 + array2 // 2

        def add_divide_by_4(array1, array2):
            return add_divide_by_2(array1, array2) // 2

    # Create convenient views
    # These views have for their first two indicies the pixels "mega pixels"
    # that contain something like
    # rg
    # gb
    # The last two indicies are the index of the subpixel within it
    red_image = color_image[:, :, 0]
    red_image.shape = (red_image.shape[0] // 2, 2, red_image.shape[1] // 2, 2)
    red_image = np.swapaxes(red_image, 1, 2)

    green_image = color_image[:, :, 1]
    green_image.shape = (
        raw_image.shape[0] // 2, 2, raw_image.shape[1] // 2, 2)
    green_image = np.swapaxes(green_image, 1, 2)

    blue_image = color_image[:, :, 2]
    blue_image.shape = (raw_image.shape[0] // 2, 2, raw_image.shape[1] // 2, 2)
    blue_image = np.swapaxes(blue_image, 1, 2)

    # TODO: allow convert to take in the "output" image
    #       this helps for large arrays, but maybe for small arrays too
    # convert(raw_image[0::2, 0::2], output=red_image[:, :, 0, 0])
    # convert(raw_image[1::2, 1::2], output=blue_image[:, :, 1, 1])
    # convert(raw_image[0::2, 1::2], output=green_image[:, :, 0, 1])
    # convert(raw_image[1::2, 0::2], output=green_image[:, :, 1, 0])

    def infill_red_or_blue_00(rb, raw_image):
        rb[:, :, 0, 0] = convert(raw_image[0::2, 0::2], dtype=dtype)
        # Compute this one first, because if the array is C continuous, this
        # Each line here is on the same cache line
        # Adjacent pixels
        rb[:, :-1, 0, 1] = add_divide_by_2(rb[:, :-1, 0, 0], rb[:, 1:, 0, 0])
        rb[:, -1, 0, 1] = rb[:, -1, 0, 0]

        # This actually takes care of the "corner" pixel because
        # The values around that one pixel have now been filled in
        rb[:-1, :, 1, :] = add_divide_by_2(rb[:-1, :, 0, :], rb[1:, :, 0, :])
        rb[-1, :, 1, :] = rb[-1, :, 0, :]

    def infill_green_00(g, raw_image):
        g[:, :, 0, 0] = convert(raw_image[0::2, 0::2], dtype=dtype)
        g[:, :, 1, 1] = convert(raw_image[1::2, 1::2], dtype=dtype)
        g[:, 1:, 1, 0] = add_divide_by_4(g[:, :-1, 1, 1], g[:, 1:, 1, 1])
        g[:, 0, 1, 0] = divide_by_2(g[:, 0, 1, 1])

        g[:, -1, 0, 1] = divide_by_2(g[:, -1, 0, 0])
        g[:, :-1, 0, 1] = add_divide_by_4(g[:, :-1, 0, 0], g[:, 1:, 0, 0])

        g[0, :, 0, 1] += divide_by_2(g[0, :, 1, 1])
        g[-1, :, 1, 0] += divide_by_2(g[-1, :, 0, 0])

        g[1:, :, 0, 1] += add_divide_by_4(g[1:, :, 1, 1], g[:-1, :, 1, 1])
        g[:-1, :, 1, 0] += add_divide_by_4(g[:-1, :, 0, 0], g[1:, :, 0, 0])

    if bayer_pattern[0] == 'r':
        infill_red_or_blue_00(red_image, raw_image)
    elif bayer_pattern[1] == 'r':
        infill_red_or_blue_00(red_image[:, ::-1, :, ::-1], raw_image[:, ::-1])
    elif bayer_pattern[2] == 'r':
        infill_red_or_blue_00(red_image[::-1, :, ::-1, :], raw_image[::-1, :])
    elif bayer_pattern[3] == 'r':
        infill_red_or_blue_00(red_image[::-1, ::-1, ::-1, ::-1],
                              raw_image[::-1, ::-1])

    if bayer_pattern[0] == 'b':
        infill_red_or_blue_00(blue_image, raw_image)
    elif bayer_pattern[1] == 'b':
        infill_red_or_blue_00(blue_image[:, ::-1, :, ::-1], raw_image[:, ::-1])
    elif bayer_pattern[2] == 'b':
        infill_red_or_blue_00(blue_image[::-1, :, ::-1, :], raw_image[::-1, :])
    elif bayer_pattern[3] == 'b':
        infill_red_or_blue_00(blue_image[::-1, ::-1, ::-1, ::-1],
                              raw_image[::-1, ::-1])
    if bayer_pattern[0] == 'g':
        infill_green_00(green_image, raw_image)
    elif bayer_pattern[1] == 'g':
        infill_green_00(green_image[::-1, :, ::-1, :], raw_image[::-1, :])

    return color_image


def bayer2rgb_filter1d(raw_image, bayer_pattern='rggb', dtype=None, *,
                       out=None):
    """This was an earlier, much cleaner implementation I had written.
    I was having trouble with the ndfilter. With the testing suite being
    corrrect now, it was easier to debug.
    I think to have gotten it now.

    I don't know which of the 3 is the fastest. Need to benchmark.

    """
    (raw_image, bayer_pattern, dtype, out) = _bayer2rgb_check_inputs(
        raw_image, bayer_pattern, dtype, out)
    # These functions are defined so as to allow floating pointers to use
    # True divide, while allowing integer types to floor divide and then
    # add avoiding overflow errors
    if dtype.kind == 'f':
        def add_divide_by_2(array1, array2):
            return (array1 + array2) * np.array(0.5, dtype=dtype)
    else:
        def add_divide_by_2(array1, array2):
            return array1 // 2 + array2 // 2

    red = out[:, :, 0]
    green = out[:, :, 1]
    blue = out[:, :, 2]

    def infill_red_or_blue_00(rb, raw_image):
        rb[::2, ::2] = convert(raw_image[0::2, 0::2], dtype=dtype)
        # Compute this one first, because if the array is C continuous, this
        # Each line here is on the same cache line
        # Adjacent pixels
        uniform_filter1d(rb[::2, ::2], size=2, origin=-1, output=rb[::2, 1::2])  # noqa
        uniform_filter1d(rb[::2, :], size=2, axis=0, origin=-1, output=rb[1::2, :])  # noqa

    def infill_green_00(green, raw_image):
        green[::2, ::2] = convert(raw_image[0::2, 0::2], dtype=dtype)
        green[1::2, 1::2] = convert(raw_image[1::2, 1::2], dtype=dtype)
        g01_from_00 = uniform_filter1d(green[::2, ::2], size=2, axis=1, origin=-1)  # noqa
        g01_from_11 = uniform_filter1d(green[1::2, 1::2], size=2, axis=0, origin=0)  # noqa

        g10_from_00 = uniform_filter1d(green[::2, ::2], size=2, axis=0, origin=-1)  # noqa
        g10_from_11 = uniform_filter1d(green[1::2, 1::2], size=2, axis=1, origin=0)  # noqa

        green[::2, 1::2] = add_divide_by_2(g01_from_00, g01_from_11)
        green[1::2, ::2] = add_divide_by_2(g10_from_00, g10_from_11)

    if bayer_pattern[0] == 'r':
        infill_red_or_blue_00(red, raw_image)
    elif bayer_pattern[1] == 'r':
        infill_red_or_blue_00(red[:, ::-1], raw_image[:, ::-1])
    elif bayer_pattern[2] == 'r':
        infill_red_or_blue_00(red[::-1, :], raw_image[::-1, :])
    elif bayer_pattern[3] == 'r':
        infill_red_or_blue_00(red[::-1, ::-1], raw_image[::-1, ::-1])

    if bayer_pattern[0] == 'b':
        infill_red_or_blue_00(blue, raw_image)
    elif bayer_pattern[1] == 'b':
        infill_red_or_blue_00(blue[:, ::-1], raw_image[:, ::-1])
    elif bayer_pattern[2] == 'b':
        infill_red_or_blue_00(blue[::-1, :], raw_image[::-1, :])
    elif bayer_pattern[3] == 'b':
        infill_red_or_blue_00(blue[::-1, ::-1], raw_image[::-1, ::-1])

    if bayer_pattern[0] == 'g':
        infill_green_00(green, raw_image)
    elif bayer_pattern[1] == 'g':
        infill_green_00(green[::-1, :], raw_image[::-1, :])

    return out


# This is a convenience dictionary primarily used
# for testing and benchmarking
implementations = {'default': bayer2rgb,
                   'slicing': bayer2rgb_slicing,
                   'filter1d': bayer2rgb_filter1d,
                   'naive': bayer2rgb_naive,
                   }
