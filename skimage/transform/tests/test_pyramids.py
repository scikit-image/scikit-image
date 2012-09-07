from numpy.testing import assert_array_equal, run_module_suite
from skimage import data
from skimage.transform import (pyramid_reduce, pyramid_expand,
                               build_gaussian_pyramid, build_laplacian_pyramid)


image = data.lena()


def test_pyramid_reduce():
    rows, cols, dim = image.shape
    out = pyramid_reduce(image, downscale=2)
    assert_array_equal(out.shape, (rows / 2, cols / 2, dim))


def test_pyramid_expand():
    rows, cols, dim = image.shape
    out = pyramid_expand(image, upscale=2)
    assert_array_equal(out.shape, (rows * 2, cols * 2, dim))


def test_build_gaussian_pyramid():
    rows, cols, dim = image.shape
    pyramid = build_gaussian_pyramid(image, downscale=2)

    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2 ** layer, cols / 2 ** layer, dim)
        assert_array_equal(out.shape, layer_shape)


def test_build_laplacian_pyramid():
    rows, cols, dim = image.shape
    pyramid = build_laplacian_pyramid(image, downscale=2)

    for layer, out in enumerate(pyramid):
        layer += 1
        layer_shape = (rows / 2 ** layer, cols / 2 ** layer, dim)
        assert_array_equal(out.shape, layer_shape)


if __name__ == "__main__":
    run_module_suite()
