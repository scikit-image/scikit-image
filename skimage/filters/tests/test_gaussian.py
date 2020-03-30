import pytest

import numpy as np
from skimage.filters._gaussian import (gaussian, _guess_spatial_dimensions,
                                       difference_of_gaussians)
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings


def test_negative_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    with testing.raises(ValueError):
        gaussian(a, sigma=-1.0)
    with testing.raises(ValueError):
        gaussian(a, sigma=[-1.0, 1.0])
    with testing.raises(ValueError):
        gaussian(a,
                 sigma=np.asarray([-1.0, 1.0]))


def test_null_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    assert np.all(gaussian(a, 0) == a)


def test_default_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    assert np.all(gaussian(a) == gaussian(a, sigma=1))


def test_energy_decrease():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    gaussian_a = gaussian(a, sigma=1, mode='reflect')
    assert gaussian_a.std() < a.std()


def test_multichannel():
    a = np.zeros((5, 5, 3))
    a[1, 1] = np.arange(1, 4)
    gaussian_rgb_a = gaussian(a, sigma=1, mode='reflect',
                              multichannel=True)
    # Check that the mean value is conserved in each channel
    # (color channels are not mixed together)
    assert np.allclose([a[..., i].mean() for i in range(3)],
                       [gaussian_rgb_a[..., i].mean() for i in range(3)])
    # Test multichannel = None
    with expected_warnings(['multichannel']):
        gaussian_rgb_a = gaussian(a, sigma=1, mode='reflect')
    # Check that the mean value is conserved in each channel
    # (color channels are not mixed together)
    assert np.allclose([a[..., i].mean() for i in range(3)],
                       [gaussian_rgb_a[..., i].mean() for i in range(3)])
    # Iterable sigma
    gaussian_rgb_a = gaussian(a, sigma=[1, 2], mode='reflect',
                              multichannel=True)
    assert np.allclose([a[..., i].mean() for i in range(3)],
                       [gaussian_rgb_a[..., i].mean() for i in range(3)])


def test_preserve_range():
    img = np.array([[10.0, -10.0], [-4, 3]], dtype=np.float32)
    gaussian(img, 1, preserve_range=True)


def test_4d_ok():
    img = np.zeros((5,) * 4)
    img[2, 2, 2, 2] = 1
    res = gaussian(img, 1, mode='reflect')
    assert np.allclose(res.sum(), 1)


def test_guess_spatial_dimensions():
    im1 = np.zeros((5, 5))
    im2 = np.zeros((5, 5, 5))
    im3 = np.zeros((5, 5, 3))
    im4 = np.zeros((5, 5, 5, 3))
    im5 = np.zeros((5,))
    testing.assert_equal(_guess_spatial_dimensions(im1), 2)
    testing.assert_equal(_guess_spatial_dimensions(im2), 3)
    testing.assert_equal(_guess_spatial_dimensions(im3), None)
    testing.assert_equal(_guess_spatial_dimensions(im4), 3)
    with testing.raises(ValueError):
        _guess_spatial_dimensions(im5)


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64]
)
def test_preserve_output(dtype):
    image = np.arange(9, dtype=dtype).reshape((3, 3))
    output = np.zeros_like(image, dtype=dtype)
    gaussian_image = gaussian(image, sigma=1, output=output,
                              preserve_range=True)
    assert gaussian_image is output


def test_output_error():
    image = np.arange(9, dtype=np.float32).reshape((3, 3))
    output = np.zeros_like(image, dtype=np.uint8)
    with pytest.raises(ValueError):
        gaussian(image, sigma=1, output=output,
                 preserve_range=True)


@testing.parametrize("s", [1, (2, 3)])
@testing.parametrize("s2", [4, (5, 6)])
def test_difference_of_gaussians(s, s2):
    image = np.random.rand(10, 10)
    im1 = gaussian(image, s)
    im2 = gaussian(image, s2)
    dog = im1 - im2
    dog2 = difference_of_gaussians(image, s, s2)
    assert np.allclose(dog, dog2)


@testing.parametrize("s", [1, (1, 2)])
def test_auto_sigma2(s):
    image = np.random.rand(10, 10)
    im1 = gaussian(image, s)
    s2 = 1.6 * np.array(s)
    im2 = gaussian(image, s2)
    dog = im1 - im2
    dog2 = difference_of_gaussians(image, s, s2)
    assert np.allclose(dog, dog2)


def test_dog_invalid_sigma_dims():
    image = np.ones((5, 5, 3))
    with testing.raises(ValueError):
        difference_of_gaussians(image, (1, 2))
    with testing.raises(ValueError):
        difference_of_gaussians(image, 1, (3, 4))
    with testing.raises(ValueError):
        difference_of_gaussians(image, (1, 2, 3), multichannel=True)


def test_dog_invalid_sigma2():
    image = np.ones((3, 3))
    with testing.raises(ValueError):
        difference_of_gaussians(image, 3, 2)
    with testing.raises(ValueError):
        difference_of_gaussians(image, (1, 5), (2, 4))
