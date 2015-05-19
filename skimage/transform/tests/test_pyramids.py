from numpy.testing import assert_array_equal, run_module_suite
import pytest

import numpy as np
from skimage import data
from skimage.transform import pyramids


image = data.astronaut()
image_gray = image[..., 0]


def test_pyramid_reduce_rgb():
    rows, cols, dim = image.shape
    out = pyramids.pyramid_reduce(image, downscale=2)
    assert_array_equal(out.shape, (rows / 2, cols / 2, dim))


def test_pyramid_reduce_gray():
    rows, cols = image_gray.shape
    out = pyramids.pyramid_reduce(image_gray, downscale=2)
    assert_array_equal(out.shape, (rows / 2, cols / 2))


def test_pyramid_reduce_nd():
    for ndim in [1, 2, 3, 4]:
        img = np.random.randn(*((8, ) * ndim))
        out = pyramids.pyramid_reduce(img, downscale=2,
                                      multichannel=False)
        expected_shape = np.asarray(img.shape) / 2
        assert_array_equal(out.shape, expected_shape)


def test_pyramid_expand_rgb():
    rows, cols, dim = image.shape
    out = pyramids.pyramid_expand(image, upscale=2)
    assert_array_equal(out.shape, (rows * 2, cols * 2, dim))


def test_pyramid_expand_gray():
    rows, cols = image_gray.shape
    out = pyramids.pyramid_expand(image_gray, upscale=2)
    assert_array_equal(out.shape, (rows * 2, cols * 2))


def test_pyramid_expand_nd():
    for ndim in [1, 2, 3, 4]:
        img = np.random.randn(*((4, ) * ndim))
        out = pyramids.pyramid_expand(img, upscale=2,
                                      multichannel=False)
        expected_shape = np.asarray(img.shape) * 2
        assert_array_equal(out.shape, expected_shape)


def test_build_gaussian_pyramid_rgb():
    rows, cols, dim = image.shape
    pyramid = pyramids.pyramid_gaussian(image, downscale=2)
    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2 ** layer, cols / 2 ** layer, dim)
        assert_array_equal(out.shape, layer_shape)


def test_build_gaussian_pyramid_gray():
    rows, cols = image_gray.shape
    pyramid = pyramids.pyramid_gaussian(image_gray, downscale=2)
    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2 ** layer, cols / 2 ** layer)
        assert_array_equal(out.shape, layer_shape)


def test_build_gaussian_pyramid_nd():
    for ndim in [1, 2, 3, 4]:
        img = np.random.randn(*((8, ) * ndim))
        original_shape = np.asarray(img.shape)
        pyramid = pyramids.pyramid_gaussian(img, downscale=2,
                                            multichannel=False)
        for layer, out in enumerate(pyramid):
            layer_shape = original_shape / 2 ** layer
            assert_array_equal(out.shape, layer_shape)


def test_build_laplacian_pyramid_rgb():
    rows, cols, dim = image.shape
    pyramid = pyramids.pyramid_laplacian(image, downscale=2)
    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2 ** layer, cols / 2 ** layer, dim)
        assert_array_equal(out.shape, layer_shape)


def test_build_laplacian_pyramid_nd():
    for ndim in [1, 2, 3, 4]:
        img = np.random.randn(*((8, ) * ndim))
        original_shape = np.asarray(img.shape)
        pyramid = pyramids.pyramid_laplacian(img, downscale=2,
                                             multichannel=False)
        for layer, out in enumerate(pyramid):
            layer_shape = original_shape / 2 ** layer
            assert_array_equal(out.shape, layer_shape)


def test_check_factor():
    with pytest.raises(ValueError):
        pyramids._check_factor(0.99)
    with pytest.raises(ValueError):
        pyramids._check_factor(- 2)


if __name__ == "__main__":
    run_module_suite()
