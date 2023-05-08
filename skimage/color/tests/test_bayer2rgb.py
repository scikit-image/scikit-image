import numpy as np
from skimage._shared.testing import parametrize, raises
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import assert_almost_equal
from skimage._shared.testing import assert_allclose

from skimage.util.dtype import convert

# Internal function used to test mathematical behavior
# Easy implementation, but slow
from skimage.color.bayer2rgb import implementations
import pytest


"""
Note to developers:

I (Mark) don't think that there is a ground truth for converting between
raw bayer images and RGB.

In fact, in my personal tests, I don't think opencv even passes these strict
tests.
They get close, but definitely fail at getting the correct values on the edges.
"""


@pytest.fixture(params=list(implementations.keys()))
def bayer2rgb(request):
    return implementations[request.param]


def test_bayer2rgb_bogus_pattern(bayer2rgb):
    # Bogus pattern
    with raises(ValueError):
        bayer2rgb(np.zeros((2, 2)), 'gggg')


@parametrize("shape", [(3, 4), (4, 3), (3, 3)])
def test_bayer2rgb_bogus_shape(bayer2rgb, shape):
    # image of odd shape
    with raises(ValueError):
        bayer2rgb(np.zeros(shape=shape))


def test_bayer2rgb_valid_inputs(bayer2rgb):
    bayer_image = np.array([[1, 0.5], [0.25, 0.33]], dtype=float)
    bayer2rgb(bayer_image, dtype='float32')
    bayer2rgb(bayer_image, dtype=np.float32)
    bayer2rgb(bayer_image, dtype=np.float)
    bayer2rgb(bayer_image, dtype=np.dtype('float32'))


def helper_test_debayer(bayer2rgb, bayer_image, expected, pattern, dtype):
    if dtype != 'float64':
        with expected_warnings(['precision loss']):
            b = convert(bayer_image, dtype=dtype)
        with expected_warnings(['precision loss']):
            e = convert(expected, dtype=dtype)
    else:
        b = convert(bayer_image, dtype=dtype)
        e = convert(expected, dtype=dtype)
    color_image = bayer2rgb(b, pattern)
    if b.dtype.kind == 'f':
        assert_almost_equal(e[..., 0], color_image[..., 0])
        assert_almost_equal(e[..., 1], color_image[..., 1])
        assert_almost_equal(e[..., 2], color_image[..., 2])
    else:
        # We divide by 4, therefore, we might be off by as
        # much as 4???
        assert_allclose(e[..., 0], color_image[..., 0], atol=4)
        assert_allclose(e[..., 1], color_image[..., 1], atol=4)
        assert_allclose(e[..., 2], color_image[..., 2], atol=4)


@parametrize("dtype", ['float64', 'float32', 'uint16',
                       'uint8', 'int16', 'uint8'])
def test_bayer2rgb_2x2(bayer2rgb, dtype):
    """A very simple test for debayering, only contains 1 super pixel."""

    # The tests compute the color image in a rather simple, brute force way.
    # Symmetries could be used, but that would complicate the logic
    # and hinder the tests. Symmetries are already used in the
    # functions themselves
    bayer_image = np.array([[1, 0.5], [0.25, 0.33]], dtype=float)

    # edge case 2x2 pixel containing only "one" super pixel
    # grbg
    expected_color_image = np.empty(
        (bayer_image.shape[0], bayer_image.shape[1], 3),
        dtype=bayer_image.dtype)
    expected_color_image[:, :, 0] = bayer_image[0, 1]
    expected_color_image[:, :, 2] = bayer_image[1, 0]
    expected_color_image[:, :, 1] = (bayer_image[0, 0] + bayer_image[1, 1]) / 2
    expected_color_image[0, 0, 1] = bayer_image[0, 0]
    expected_color_image[1, 1, 1] = bayer_image[1, 1]

    helper_test_debayer(bayer2rgb,
                        bayer_image, expected_color_image, 'grbg', dtype)

    # gbrg
    expected_color_image[..., 2], expected_color_image[..., 0] = \
        expected_color_image[..., 0].copy(), expected_color_image[..., 2].copy()  # noqa
    helper_test_debayer(bayer2rgb,
                        bayer_image, expected_color_image, 'gbrg', dtype)

    # rggb
    expected_color_image[:, :, 0] = bayer_image[0, 0]
    expected_color_image[:, :, 2] = bayer_image[1, 1]
    expected_color_image[:, :, 1] = (bayer_image[0, 1] + bayer_image[1, 0]) / 2
    expected_color_image[0, 1, 1] = bayer_image[0, 1]
    expected_color_image[1, 0, 1] = bayer_image[1, 0]

    helper_test_debayer(bayer2rgb,
                        bayer_image, expected_color_image, 'rggb', dtype)

    # bggr
    expected_color_image[..., 2], expected_color_image[..., 0] = \
        expected_color_image[..., 0].copy(), expected_color_image[..., 2].copy()  # noqa

    helper_test_debayer(bayer2rgb,
                        bayer_image, expected_color_image, 'bggr', dtype)


@parametrize("dtype", ['float64', 'float32', 'uint16',
                       'uint8', 'int16', 'uint8'])
def test_bayer2rgb_4x4(bayer2rgb, dtype):
    # This is a 4x4 image sensor.
    # it tests for all cases I think. middle points, and edge points with 2
    # neighbors.
    bayer_image = np.reshape(np.arange(1, 16 + 1, dtype=float), (4, 4)) / 16

    # rggb
    expected_color_image = np.zeros((4, 4, 3), dtype=bayer_image.dtype)
    expected_color_image[0::2, 0::2, 0] = bayer_image[0::2, 0::2]
    expected_color_image[1::2, 0::2, 1] = bayer_image[1::2, 0::2]
    expected_color_image[0::2, 1::2, 1] = bayer_image[0::2, 1::2]
    expected_color_image[1::2, 1::2, 2] = bayer_image[1::2, 1::2]

    red = expected_color_image[..., 0]
    green = expected_color_image[..., 1]
    blue = expected_color_image[..., 2]

    red[(0, 2), 1] = (red[(0, 2), 0] + red[(0, 2), 2]) / 2
    red[(0, 2), 3] = red[(0, 2), 2]
    red[1, :] = (red[0, :] + red[2, :]) / 2
    red[3, :] = red[2, :]

    blue[(1, 3), 2] = (blue[(1, 3), 1] + blue[(1, 3), 3]) / 2
    blue[(1, 3), 0] = blue[(1, 3), 1]
    blue[2, :] = (blue[1, :] + blue[3, :]) / 2
    blue[0, :] = blue[1, :]

    green[0, 0] = (green[0, 1] + green[1, 0]) / 2
    green[-1, -1] = (green[-1, -2] + green[-2, -1]) / 2
    green[0, 2] = green[0, 1] * 0.25 + green[0, 3] * 0.25 + green[1, 2] * 0.5
    green[2, 0] = green[1, 0] * 0.25 + green[3, 0] * 0.25 + green[2, 1] * 0.5
    green[-1, 1] = green[-1, 0] * 0.25 + green[-1, 2] * 0.25 + green[-2, 1] * 0.5  # noqa
    green[1, -1] = green[0, -1] * 0.25 + green[2, -1] * 0.25 + green[1, -2] * 0.5  # noqa

    green[1, 1] = (green[0, 1] + green[1, 0] + green[2, 1] + green[1, 2]) / 4
    green[2, 2] = (green[1, 2] + green[2, 1] + green[3, 2] + green[2, 3]) / 4

    helper_test_debayer(bayer2rgb,
                        bayer_image, expected_color_image, 'rggb', dtype)

    # bggr
    red[...], blue[...] = blue.copy(), red.copy()
    helper_test_debayer(bayer2rgb,
                        bayer_image, expected_color_image, 'bggr', dtype)

    # gbrg
    expected_color_image = np.zeros((4, 4, 3), dtype=bayer_image.dtype)
    expected_color_image[0::2, 0::2, 1] = bayer_image[0::2, 0::2]
    expected_color_image[1::2, 0::2, 0] = bayer_image[1::2, 0::2]
    expected_color_image[0::2, 1::2, 2] = bayer_image[0::2, 1::2]
    expected_color_image[1::2, 1::2, 1] = bayer_image[1::2, 1::2]

    red = expected_color_image[..., 0]
    green = expected_color_image[..., 1]
    blue = expected_color_image[..., 2]

    red[(1, 3), 1] = (red[(1, 3), 0] + red[(1, 3), 2]) / 2
    red[(1, 3), 3] = red[(1, 3), 2]
    red[2, :] = (red[1, :] + red[3, :]) / 2
    red[0, :] = red[1, :]

    blue[(0, 2), 2] = (blue[(0, 2), 1] + blue[(0, 2), 3]) / 2
    blue[(0, 2), 0] = blue[(0, 2), 1]
    blue[1, :] = (blue[0, :] + blue[2, :]) / 2
    blue[-1, :] = blue[-2, :]

    green[0, -1] = (green[0, -2] + green[1, -1]) / 2
    green[-1, 0] = (green[-2, 0] + green[-1, 1]) / 2
    green[0, 1] = green[0, 0] * 0.25 + green[0, 2] * 0.25 + green[1, 1] * 0.5
    green[1, 0] = green[0, 0] * 0.25 + green[2, 0] * 0.25 + green[1, 1] * 0.5

    green[-1, 2] = green[-1, 1] * 0.25 + green[-1, 3] * 0.25 + green[-2, 2] * 0.5  # noqa
    green[2, -1] = green[1, -1] * 0.25 + green[3, -1] * 0.25 + green[2, -2] * 0.5  # noqa

    green[2, 1] = (green[1, 1] + green[3, 1] + green[2, 2] + green[2, 0]) * 0.25  # noqa
    green[1, 2] = (green[1, 1] + green[1, 3] + green[2, 2] + green[0, 2]) * 0.25  # noqa

    helper_test_debayer(bayer2rgb,
                        bayer_image, expected_color_image, 'gbrg', dtype)

    # grbg
    red[...], blue[...] = blue.copy(), red.copy()
    helper_test_debayer(bayer2rgb,
                        bayer_image, expected_color_image, 'grbg', dtype)
