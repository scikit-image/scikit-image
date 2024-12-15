import numpy as np
import pytest

import skimage as ski


def test_blur_effect():
    """Test that the blur metric increases with more blurring."""
    image = ski.data.astronaut()
    B0 = ski.measure.blur_effect(image, channel_axis=-1)
    B1 = ski.measure.blur_effect(
        ski.filters.gaussian(image, sigma=1, channel_axis=-1), channel_axis=-1
    )
    B2 = ski.measure.blur_effect(
        ski.filters.gaussian(image, sigma=4, channel_axis=-1), channel_axis=-1
    )
    assert 0 <= B0 < 1
    assert B0 < B1 < B2


def test_blur_effect_h_size():
    """Test that the blur metric decreases with increasing size of the
    re-blurring filter.
    """
    image = ski.data.astronaut()
    B0 = ski.measure.blur_effect(image, h_size=3, channel_axis=-1)
    B1 = ski.measure.blur_effect(image, channel_axis=-1)  # default h_size is 11
    B2 = ski.measure.blur_effect(image, h_size=30, channel_axis=-1)
    assert 0 <= B0 < 1
    assert B0 > B1 > B2


def test_blur_effect_channel_axis():
    """Test that passing an RGB image is equivalent to passing its grayscale
    version.
    """
    image = ski.data.astronaut()
    B0 = ski.measure.blur_effect(image, channel_axis=-1)
    B1 = ski.measure.blur_effect(ski.color.rgb2gray(image))
    B0_arr = ski.measure.blur_effect(image, channel_axis=-1, reduce_func=None)
    B1_arr = ski.measure.blur_effect(ski.color.rgb2gray(image), reduce_func=None)
    assert 0 <= B0 < 1
    assert B0 == B1
    np.testing.assert_array_equal(B0_arr, B1_arr)


def test_blur_effect_3d():
    """Test that the blur metric works on a 3D image."""
    image_3d = ski.data.cells3d()[:, 1, :, :]  # grab just the nuclei
    B0 = ski.measure.blur_effect(image_3d)
    B1 = ski.measure.blur_effect(ski.filters.gaussian(image_3d, sigma=1))
    B2 = ski.measure.blur_effect(ski.filters.gaussian(image_3d, sigma=4))
    assert 0 <= B0 < 1
    assert B0 < B1 < B2


@pytest.mark.parametrize('image', [np.zeros((100, 100, 3)), np.ones((100, 100, 3))])
def test_blur_effect_uniform_input(image):
    """Test that the blur metric is 1 for completely uniform images."""
    image = np.ones((100, 100, 3))
    with np.testing.assert_warns(UserWarning):
        B = ski.measure.blur_effect(image)
        assert B == 1
