from numpy.testing import assert_array_equal, assert_almost_equal, assert_equal
import numpy as np

from skimage.color import rgb2gray
from skimage.data import astronaut, cells3d
from skimage.filters import gaussian
from skimage.measure import blur_effect


def test_blur_effect():
    """Test that the blur metric increases with more blurring."""
    image = astronaut()
    B0 = blur_effect(image, channel_axis=-1)
    B1 = blur_effect(gaussian(image, sigma=1, channel_axis=-1), channel_axis=-1)
    B2 = blur_effect(gaussian(image, sigma=4, channel_axis=-1), channel_axis=-1)
    assert 0 <= B0 < 1
    assert B0 < B1 < B2


def test_blur_effect_h_size():
    """Test that the blur metric decreases with increasing size of the
    re-blurring filter.
    """
    image = astronaut()
    B0 = blur_effect(image, h_size=3, channel_axis=-1)
    B1 = blur_effect(image, channel_axis=-1)  # default h_size is 11
    B2 = blur_effect(image, h_size=30, channel_axis=-1)
    assert 0 <= B0 < 1
    assert B0 > B1 > B2


def test_blur_effect_channel_axis():
    """Test that passing an RGB image is equivalent to passing its grayscale
    version.
    """
    image = astronaut()
    B0 = blur_effect(image, channel_axis=-1)
    B1 = blur_effect(rgb2gray(image))
    B0_arr = blur_effect(image, channel_axis=-1, reduce_func=None)
    B1_arr = blur_effect(rgb2gray(image), reduce_func=None)
    assert 0 <= B0 < 1
    assert B0 == B1
    assert_array_equal(B0_arr, B1_arr)


def test_blur_effect_3d():
    """Test that the blur metric works on a 3D image."""
    image_3d = cells3d()[:, 1, :, :]  # grab just the nuclei
    B0 = blur_effect(image_3d)
    B1 = blur_effect(gaussian(image_3d, sigma=1))
    B2 = blur_effect(gaussian(image_3d, sigma=4))
    assert 0 <= B0 < 1
    assert B0 < B1 < B2


def test_blur_constant_image():
    """Test that the blur metric works for a constant image."""
    image = np.zeros((1080, 1920, 3), dtype=int)
    B0 = blur_effect(image)
    assert_equal(B0, 1.0)


def test_blur_single_axis_constant_image():
    """Test that the blur metric work for an image that is constant in one axis."""
    row = np.linspace(0, 1, 1000)
    image = np.array([row for _ in range(1000)])
    B0 = blur_effect(image)
    assert_almost_equal(B0, 1.0, decimal=5)
