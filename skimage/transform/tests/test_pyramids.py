import math

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal

from skimage import data, transform
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.transform import pyramids


image = data.astronaut()
image_gray = image[..., 0]


@pytest.mark.parametrize('channel_axis', [0, 1, -1])
def test_pyramid_reduce_rgb(channel_axis):
    rows, cols, dim = image.shape
    image_ = np.moveaxis(image, -1, channel_axis)
    out_ = transform.pyramid_reduce(image_, downscale=2,
                                    channel_axis=channel_axis)
    out = np.moveaxis(out_, channel_axis, -1)
    assert_array_equal(out.shape, (rows / 2, cols / 2, dim))


def test_pyramid_reduce_gray():
    rows, cols = image_gray.shape
    out1 = transform.pyramid_reduce(image_gray, downscale=2, channel_axis=None)
    assert_array_equal(out1.shape, (rows / 2, cols / 2))
    assert_almost_equal(out1.ptp(), 1.0, decimal=2)
    out2 = transform.pyramid_reduce(image_gray, downscale=2,
                                    channel_axis=None, preserve_range=True)
    assert_almost_equal(out2.ptp() / image_gray.ptp(), 1.0, decimal=2)


def test_pyramid_reduce_nd():
    for ndim in [1, 2, 3, 4]:
        img = np.random.randn(*((8, ) * ndim))
        out = transform.pyramid_reduce(img, downscale=2, channel_axis=None)
        expected_shape = np.asarray(img.shape) / 2
        assert_array_equal(out.shape, expected_shape)


def test_pyramid_expand_rgb():
    rows, cols, dim = image.shape
    out = transform.pyramid_expand(image, upscale=2, channel_axis=-1)
    assert_array_equal(out.shape, (rows * 2, cols * 2, dim))


def test_pyramid_expand_gray():
    rows, cols = image_gray.shape
    out = transform.pyramid_expand(image_gray, upscale=2, channel_axis=None)
    assert_array_equal(out.shape, (rows * 2, cols * 2))


def test_pyramid_expand_nd():
    for ndim in [1, 2, 3, 4]:
        img = np.random.randn(*((4, ) * ndim))
        out = transform.pyramid_expand(img, upscale=2, channel_axis=None)
        expected_shape = np.asarray(img.shape) * 2
        assert_array_equal(out.shape, expected_shape)


def test_build_gaussian_pyramid_rgb():
    rows, cols, dim = image.shape
    pyramid = transform.pyramid_gaussian(image, downscale=2, channel_axis=-1)
    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2 ** layer, cols / 2 ** layer, dim)
        assert_array_equal(out.shape, layer_shape)


def test_build_gaussian_pyramid_gray():
    rows, cols = image_gray.shape
    pyramid = transform.pyramid_gaussian(image_gray, downscale=2,
                                         channel_axis=None)
    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2 ** layer, cols / 2 ** layer)
        assert_array_equal(out.shape, layer_shape)


def test_build_gaussian_pyramid_nd():
    for ndim in [1, 2, 3, 4]:
        img = np.random.randn(*((8, ) * ndim))
        original_shape = np.asarray(img.shape)
        pyramid = transform.pyramid_gaussian(img, downscale=2,
                                             channel_axis=None)
        for layer, out in enumerate(pyramid):
            layer_shape = original_shape / 2 ** layer
            assert_array_equal(out.shape, layer_shape)


def test_build_laplacian_pyramid_rgb():
    rows, cols, dim = image.shape
    pyramid = transform.pyramid_laplacian(image, downscale=2,
                                          channel_axis=-1)
    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2 ** layer, cols / 2 ** layer, dim)
        assert_array_equal(out.shape, layer_shape)


def test_build_laplacian_pyramid_nd():
    for ndim in [1, 2, 3, 4]:
        img = np.random.randn(*(16, )*ndim)
        original_shape = np.asarray(img.shape)
        pyramid = transform.pyramid_laplacian(img, downscale=2,
                                              channel_axis=None)
        for layer, out in enumerate(pyramid):
            print(out.shape)
            layer_shape = original_shape / 2 ** layer
            assert_array_equal(out.shape, layer_shape)


def test_laplacian_pyramid_max_layers():
    for downscale in [2, 3, 5, 7]:
        img = np.random.randn(32, 8)
        pyramid = transform.pyramid_laplacian(img, downscale=downscale,
                                              channel_axis=None)
        max_layer = int(np.ceil(math.log(np.max(img.shape), downscale)))
        for layer, out in enumerate(pyramid):
            if layer < max_layer:
                # should not reach all axes as size 1 prior to final level
                assert np.max(out.shape) > 1

        # total number of images is max_layer + 1
        assert_equal(max_layer, layer)

        # final layer should be size 1 on all axes
        assert_array_equal((out.shape), (1, 1))


def test_check_factor():
    with pytest.raises(ValueError):
        pyramids._check_factor(0.99)
    with pytest.raises(ValueError):
        pyramids._check_factor(- 2)


@pytest.mark.parametrize(
    'dtype', ['float16', 'float32', 'float64', 'uint8', 'int64']
)
@pytest.mark.parametrize(
    'pyramid_func', [transform.pyramid_gaussian, transform.pyramid_laplacian]
)
def test_pyramid_dtype_support(pyramid_func, dtype):
    img = np.random.randn(32, 8).astype(dtype)
    pyramid = pyramid_func(img)

    float_dtype = _supported_float_type(dtype)
    assert np.all([im.dtype == float_dtype for im in pyramid])
