"""
Tests for Rolling Ball Filter
(skimage.morphology.rolling_ball)
"""

import numpy as np
import pytest

from skimage import data
from skimage.morphology import rolling_ball


@pytest.mark.parametrize("radius", [1, 2.5, 10.346, 50, 100])
def test_const_image(radius):
    # infinite plane light source at top left corner
    img = 23 * np.ones((100, 100), dtype=np.uint8)
    result = rolling_ball(img, radius, np.max(img))
    assert np.allclose(result, np.zeros_like(img))


def test_radial_gradient():
    # spot light source at top left corner
    spot_radius = 50
    x, y = np.meshgrid(range(5), range(5))
    img = np.sqrt(np.clip(spot_radius ** 2 - y ** 2 - x ** 2, 0, None))

    result = rolling_ball(img, radius=5, max_intensity=np.max(img))
    assert np.allclose(result, np.zeros_like(img))


def test_linear_gradient():
    # linear light source centered at top left corner
    x, y = np.meshgrid(range(100), range(100))
    img = (y * 20 + x * 20)

    expected_img = 19 * np.ones_like(img)
    expected_img[0, 0] = 0

    result = rolling_ball(img, radius=1, max_intensity=np.max(img))
    assert np.allclose(result, expected_img)


@pytest.mark.parametrize("radius", [2, 10, 12.5, 50, 100])
def test_preserve_peaks(radius):
    x, y = np.meshgrid(range(100), range(100))
    img = 0 * x + 0 * y + 10
    img[10, 10] = 20
    img[20, 20] = 35
    img[45, 26] = 156

    expected_img = img - 10
    result = rolling_ball(img, radius, max_intensity=np.max(img))
    assert np.allclose(result, expected_img)
