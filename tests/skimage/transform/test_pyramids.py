import math
import warnings

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal

from skimage import data
from skimage._shared.utils import _supported_float_type
from skimage.transform import pyramids
from skimage.metrics import structural_similarity


image = data.astronaut()
image_gray = image[..., 0]


@pytest.mark.parametrize('channel_axis', [0, 1, -1])
def test_pyramid_reduce_rgb(channel_axis):
    image = data.astronaut()
    rows, cols, dim = image.shape
    image = np.moveaxis(image, source=-1, destination=channel_axis)
    out1_ = pyramids.pyramid_reduce(image, downscale=2, channel_axis=channel_axis)
    out1 = np.moveaxis(out1_, channel_axis, -1)
    assert_array_equal(out1.shape, (rows / 2, cols / 2, dim))
    assert_almost_equal(np.ptp(out1), 1.0, decimal=2)
    out2_ = pyramids.pyramid_reduce(
        image, downscale=2, channel_axis=channel_axis, preserve_range=True
    )
    out2 = np.moveaxis(out2_, channel_axis, -1)
    assert_almost_equal(np.ptp(out2) / np.ptp(image), 1.0, decimal=2)


def test_pyramid_reduce_gray():
    rows, cols = image_gray.shape
    out1 = pyramids.pyramid_reduce(image_gray, downscale=2, channel_axis=None)
    assert_array_equal(out1.shape, (rows / 2, cols / 2))
    assert_almost_equal(np.ptp(out1), 1.0, decimal=2)
    out2 = pyramids.pyramid_reduce(
        image_gray, downscale=2, channel_axis=None, preserve_range=True
    )
    assert_almost_equal(np.ptp(out2) / np.ptp(image_gray), 1.0, decimal=2)


def test_pyramid_reduce_gray_defaults():
    rows, cols = image_gray.shape
    out1 = pyramids.pyramid_reduce(image_gray)
    assert_array_equal(out1.shape, (rows / 2, cols / 2))
    assert_almost_equal(np.ptp(out1), 1.0, decimal=2)
    out2 = pyramids.pyramid_reduce(image_gray, preserve_range=True)
    assert_almost_equal(np.ptp(out2) / np.ptp(image_gray), 1.0, decimal=2)


@pytest.mark.parametrize('mode', ['reflect', 'constant', 'edge', 'symmetric', 'wrap'])
def test_pyramid_reduce_modes(mode):
    rows, cols = image_gray.shape
    out1 = pyramids.pyramid_reduce(image_gray, mode=mode)
    assert_array_equal(out1.shape, (rows / 2, cols / 2))
    assert_almost_equal(np.ptp(out1), 1.0, decimal=2)
    out2 = pyramids.pyramid_reduce(image_gray, mode=mode, preserve_range=True)
    assert_almost_equal(np.ptp(out2) / np.ptp(image_gray), 1.0, decimal=2)


def test_pyramid_reduce_nd():
    for ndim in [1, 2, 3, 4]:
        img = np.random.randn(*((8,) * ndim))
        out1 = pyramids.pyramid_reduce(img, downscale=2, channel_axis=None)
        expected_shape = np.asarray(img.shape) / 2
        assert_array_equal(out1.shape, expected_shape)
        # no ptp check because it's a small array of random numbers


@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1, -2, -3])
def test_pyramid_expand_rgb(channel_axis):
    image = data.astronaut()
    rows, cols, dim = image.shape
    image = np.moveaxis(image, source=-1, destination=channel_axis)
    out1 = pyramids.pyramid_expand(image, upscale=2, channel_axis=channel_axis)
    expected_shape = [rows * 2, cols * 2]
    expected_shape.insert(channel_axis % image.ndim, dim)
    assert_array_equal(out1.shape, expected_shape)
    assert_almost_equal(np.ptp(out1), 1.0, decimal=2)
    out2 = pyramids.pyramid_expand(
        image, upscale=2, channel_axis=channel_axis, preserve_range=True
    )
    assert_almost_equal(np.ptp(out2) / np.ptp(image), 1.0, decimal=2)


def test_pyramid_expand_gray():
    rows, cols = image_gray.shape
    out1 = pyramids.pyramid_expand(image_gray, upscale=2)
    assert_array_equal(out1.shape, (rows * 2, cols * 2))
    assert_almost_equal(np.ptp(out1), 1.0, decimal=2)
    out2 = pyramids.pyramid_expand(image_gray, upscale=2, preserve_range=True)
    assert_almost_equal(np.ptp(out2) / np.ptp(image_gray), 1.0, decimal=2)


@pytest.mark.parametrize('mode', ['reflect', 'constant', 'edge', 'symmetric', 'wrap'])
def test_pyramid_expand_modes(mode):
    rows, cols = image_gray.shape
    out1 = pyramids.pyramid_expand(image_gray, mode=mode)
    assert_array_equal(out1.shape, (rows * 2, cols * 2))
    assert_almost_equal(np.ptp(out1), 1.0, decimal=2)
    out2 = pyramids.pyramid_expand(image_gray, mode=mode, preserve_range=True)
    assert_almost_equal(np.ptp(out2) / np.ptp(image_gray), 1.0, decimal=2)


def test_pyramid_expand_nd():
    for ndim in [1, 2, 3, 4]:
        img = np.random.randn(*((4,) * ndim))
        out1 = pyramids.pyramid_expand(img, upscale=2, channel_axis=None)
        expected_shape = np.asarray(img.shape) * 2
        assert_array_equal(out1.shape, expected_shape)
        # no ptp check because it's a small array of random numbers


@pytest.mark.parametrize('mode', ['reflect', 'constant', 'edge', 'symmetric', 'wrap'])
def test_pyramid_expand_reduce_roundtrip(mode):
    expanded = pyramids.pyramid_expand(image_gray, mode=mode, preserve_range=True)
    reconstructed = pyramids.pyramid_reduce(expanded, mode=mode, preserve_range=True)
    assert reconstructed.shape == image_gray.shape
    mssim = structural_similarity(
        reconstructed, image_gray, data_range=np.ptp(image_gray)
    )
    assert mssim > 0.95


@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1, -2, -3])
def test_build_gaussian_pyramid_rgb(channel_axis):
    image = data.astronaut()
    rows, cols, dim = image.shape
    image = np.moveaxis(image, source=-1, destination=channel_axis)
    pyramid1 = list(
        pyramids.pyramid_gaussian(image, downscale=2, channel_axis=channel_axis)
    )
    pyramid2 = list(
        pyramids.pyramid_gaussian(
            image, downscale=2, channel_axis=channel_axis, preserve_range=True
        )
    )
    for layer, (out1, out2) in enumerate(zip(pyramid1, pyramid2, strict=True)):
        layer_shape = [rows / 2**layer, cols / 2**layer]
        layer_shape.insert(channel_axis % image.ndim, dim)
        assert out1.shape == tuple(layer_shape)
        # smaller layers are too smooth, reducing the PTP
        if max(layer_shape) >= 128:
            decimal = 2
        elif max(layer_shape) >= 16:
            decimal = 1
        else:
            decimal = 0
        assert_almost_equal(np.ptp(out1), 1.0, decimal=decimal, err_msg=f"layer {layer} shape {out1.shape}")
        assert_almost_equal(np.ptp(out2) / np.ptp(image), 1.0, decimal=decimal, err_msg=f"layer {layer} shape {out2.shape}")


def test_build_gaussian_pyramid_gray():
    rows, cols = image_gray.shape
    pyramid1 = list(
        pyramids.pyramid_gaussian(image_gray, downscale=2, channel_axis=None)
    )
    pyramid2 = list(
        pyramids.pyramid_gaussian(
            image_gray, downscale=2, channel_axis=None, preserve_range=True
        )
    )
    for layer, (out1, out2) in enumerate(zip(pyramid1, pyramid2, strict=True)):
        layer_shape = (rows / 2**layer, cols / 2**layer)
        assert_array_equal(out1.shape, layer_shape)
        # smaller layers are too smooth, reducing the PTP
        if max(layer_shape) >= 128:
            decimal = 2
        elif max(layer_shape) >= 16:
            decimal = 1
        else:
            decimal = 0
        assert_almost_equal(np.ptp(out1), 1.0, decimal=decimal)
        assert_almost_equal(np.ptp(out2) / np.ptp(image_gray), 1.0, decimal=decimal)


def test_build_gaussian_pyramid_gray_defaults():
    rows, cols = image_gray.shape
    pyramid1 = list(pyramids.pyramid_gaussian(image_gray))
    pyramid2 = list(pyramids.pyramid_gaussian(image_gray, preserve_range=True))
    for layer, (out1, out2) in enumerate(zip(pyramid1, pyramid2, strict=True)):
        layer_shape = (rows / 2**layer, cols / 2**layer)
        assert_array_equal(out1.shape, layer_shape)
        # smaller layers are too smooth, reducing the PTP
        if max(layer_shape) >= 128:
            decimal = 2
        elif max(layer_shape) >= 16:
            decimal = 1
        else:
            decimal = 0
        assert_almost_equal(np.ptp(out1), 1.0, decimal=decimal)
        assert_almost_equal(np.ptp(out2) / np.ptp(image_gray), 1.0, decimal=decimal)


@pytest.mark.parametrize('mode', ['reflect', 'constant', 'edge', 'symmetric', 'wrap'])
def test_build_gaussian_pyramid_gray_modes(mode):
    rows, cols = image_gray.shape
    pyramid1 = list(pyramids.pyramid_gaussian(image_gray, mode=mode))
    pyramid2 = list(
        pyramids.pyramid_gaussian(image_gray, mode=mode, preserve_range=True)
    )
    for layer, (out1, out2) in enumerate(zip(pyramid1, pyramid2, strict=True)):
        layer_shape = (rows / 2**layer, cols / 2**layer)
        assert_array_equal(out1.shape, layer_shape)
        # smaller layers are too smooth, reducing the PTP
        if max(layer_shape) >= 128:
            decimal = 2
        elif max(layer_shape) >= 16:
            decimal = 1
        else:
            decimal = 0
        assert_almost_equal(np.ptp(out1), 1.0, decimal=decimal)
        assert_almost_equal(np.ptp(out2) / np.ptp(image_gray), 1.0, decimal=decimal)


def test_build_gaussian_pyramid_nd():
    for ndim in [1, 2, 3, 4]:
        img = np.random.randn(*((8,) * ndim))
        original_shape = np.asarray(img.shape)
        pyramid1 = list(pyramids.pyramid_gaussian(img, downscale=2, channel_axis=None))
        pyramid2 = list(
            pyramids.pyramid_gaussian(
                img, downscale=2, channel_axis=None, preserve_range=True
            )
        )
        for layer, (out1, out2) in enumerate(zip(pyramid1, pyramid2, strict=True)):
            layer_shape = original_shape / 2**layer
            assert_array_equal(out1.shape, layer_shape)
            # no ptp check because it's a small array of random numbers


@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1, -2, -3])
def test_build_laplacian_pyramid_rgb(channel_axis):
    image = data.astronaut()
    rows, cols, dim = image.shape
    image = np.moveaxis(image, source=-1, destination=channel_axis)
    pyramid1 = list(
        pyramids.pyramid_laplacian(image, downscale=2, channel_axis=channel_axis)
    )
    pyramid2 = list(
        pyramids.pyramid_laplacian(
            image, downscale=2, channel_axis=channel_axis, preserve_range=True
        )
    )
    for layer, (out1, out2) in enumerate(zip(pyramid1, pyramid2, strict=True)):
        layer_shape = [rows / 2**layer, cols / 2**layer]
        layer_shape.insert(channel_axis % image.ndim, dim)
        assert out1.shape == tuple(layer_shape)
        # smaller layers are too smooth, reducing the PTP
        if max(layer_shape) >= 128:
            decimal = 2
        elif max(layer_shape) >= 16:
            decimal = 1
        else:
            decimal = 0
        assert_almost_equal(np.ptp(out1), 1.0, decimal=decimal, err_msg=f"layer {layer} shape {out1.shape}")
        assert_almost_equal(np.ptp(out2) / np.ptp(image), 1.0, decimal=decimal, err_msg=f"layer {layer} shape {out2.shape}")


def test_build_laplacian_pyramid_defaults():
    rows, cols = image_gray.shape
    pyramid1 = list(pyramids.pyramid_laplacian(image_gray))
    pyramid2 = list(pyramids.pyramid_laplacian(image_gray, preserve_range=True))
    for layer, (out1, out2) in enumerate(zip(pyramid1, pyramid2, strict=True)):
        layer_shape = (rows / 2**layer, cols / 2**layer)
        assert_array_equal(out1.shape, layer_shape)
        assert_almost_equal(np.ptp(out1), 1.0, decimal=1)
        assert_almost_equal(np.ptp(out2) / np.ptp(image_gray), 1.0, decimal=1)


@pytest.mark.parametrize('mode', ['reflect', 'constant', 'edge', 'symmetric', 'wrap'])
def test_build_laplacian_pyramid_modes(mode):
    rows, cols = image_gray.shape
    pyramid1 = list(pyramids.pyramid_laplacian(image_gray, mode=mode))
    pyramid2 = list(
        pyramids.pyramid_laplacian(image_gray, mode=mode, preserve_range=True)
    )
    for layer, (out1, out2) in enumerate(zip(pyramid1, pyramid2, strict=True)):
        layer_shape = (rows / 2**layer, cols / 2**layer)
        assert_array_equal(out1.shape, layer_shape)
        assert_almost_equal(np.ptp(out1), 1.0, decimal=1)
        assert_almost_equal(np.ptp(out2) / np.ptp(image_gray), 1.0, decimal=1)


def test_build_laplacian_pyramid_nd():
    for ndim in [1, 2, 3, 4]:
        img = np.random.randn(*(16,) * ndim)
        original_shape = np.asarray(img.shape)
        pyramid1 = list(pyramids.pyramid_laplacian(img, downscale=2, channel_axis=None))
        pyramid2 = list(
            pyramids.pyramid_laplacian(
                img, downscale=2, channel_axis=None, preserve_range=True
            )
        )
        for layer, (out1, out2) in enumerate(zip(pyramid1, pyramid2, strict=True)):
            layer_shape = original_shape / 2**layer
            assert_array_equal(out1.shape, layer_shape)
            assert_almost_equal(np.ptp(out1), 1.0, decimal=1)
            assert_almost_equal(np.ptp(out2) / np.ptp(img), 1.0, decimal=1)


@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1, -2, -3])
def test_laplacian_pyramid_max_layers(channel_axis):
    for downscale in [2, 3, 5, 7]:
        if channel_axis is None:
            shape = (32, 8)
            shape_without_channels = shape
        else:
            shape_without_channels = (32, 8)
            ndim = len(shape_without_channels) + 1
            n_channels = 5
            shape = list(shape_without_channels)
            shape.insert(channel_axis % ndim, n_channels)
            shape = tuple(shape)
        img = np.ones(shape)
        pyramid = pyramids.pyramid_laplacian(
            img, downscale=downscale, channel_axis=channel_axis
        )
        max_layer = math.ceil(math.log(max(shape_without_channels), downscale))
        for layer, out in enumerate(pyramid):
            if channel_axis is None:
                out_shape_without_channels = out.shape
            else:
                assert out.shape[channel_axis] == n_channels
                out_shape_without_channels = list(out.shape)
                out_shape_without_channels.pop(channel_axis)
                out_shape_without_channels = tuple(out_shape_without_channels)

            if layer < max_layer:
                # should not reach all axes as size 1 prior to final level
                assert max(out_shape_without_channels) > 1

        # total number of images is max_layer + 1
        assert_equal(max_layer, layer)

        # final layer should be size 1 on all axes
        assert out_shape_without_channels == (1, 1)


def test_check_factor():
    with pytest.raises(ValueError):
        pyramids._check_factor(0.99)
    with pytest.raises(ValueError):
        pyramids._check_factor(-2)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'uint8', 'int64'])
@pytest.mark.parametrize(
    'pyramid_func', [pyramids.pyramid_gaussian, pyramids.pyramid_laplacian]
)
def test_pyramid_dtype_support(pyramid_func, dtype):
    with warnings.catch_warnings():
        # Ignore arch specific warning on arm64, armhf, ppc64el, riscv64, s390x
        # https://github.com/scikit-image/scikit-image/issues/7391
        warnings.filterwarnings(
            action="ignore",
            category=RuntimeWarning,
            message="invalid value encountered in cast",
        )
        img = np.random.randn(32, 8).astype(dtype)

    pyramid = pyramid_func(img)
    float_dtype = _supported_float_type(dtype)
    assert np.all([im.dtype == float_dtype for im in pyramid])
