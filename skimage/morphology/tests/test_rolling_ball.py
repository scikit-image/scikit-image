"""
Tests for Rolling Ball Filter
(skimage.morphology.rolling_ball)
"""

import numpy as np
import pytest

from skimage import data, io
from skimage.morphology import rolling_ball, rolling_ellipsoid


def test_ellipsoid_const():
    img = 155 * np.ones((100, 100), dtype=np.uint8)
    background = rolling_ellipsoid(img, kernel_shape=(25, 53))
    assert np.allclose(img - background, np.zeros_like(img))


def test_nan_const():
    img = 123 * np.ones((100, 100), dtype=np.float_)
    img[20, 20] = np.nan
    img[50, 53] = np.nan

    kernel_shape = (10, 10)
    x = np.arange(-kernel_shape[1] // 2,
                  kernel_shape[1] // 2 + 1)[np.newaxis, :]
    y = np.arange(-kernel_shape[0] // 2,
                  kernel_shape[0] // 2 + 1)[:, np.newaxis]
    expected_img = np.zeros_like(img)
    expected_img[y + 20, x + 20] = np.nan
    expected_img[y + 50, x + 53] = np.nan
    background = rolling_ellipsoid(
        img,
        kernel_shape=kernel_shape,
        nansafe=True
    )
    assert np.allclose(img - background, expected_img, equal_nan=True)


@pytest.mark.parametrize("radius", [1, 2.5, 10.346, 50])
def test_const_image(radius):
    # infinite plane light source at top left corner
    img = 23 * np.ones((100, 100), dtype=np.uint8)
    background = rolling_ball(img, radius)
    assert np.allclose(img - background, np.zeros_like(img))


def test_radial_gradient():
    # spot light source at top left corner
    spot_radius = 50
    x, y = np.meshgrid(range(5), range(5))
    img = np.sqrt(np.clip(spot_radius ** 2 - y ** 2 - x ** 2, 0, None))

    background = rolling_ball(img, radius=5)
    assert np.allclose(img - background, np.zeros_like(img))


def test_linear_gradient():
    # linear light source centered at top left corner
    x, y = np.meshgrid(range(100), range(100))
    img = (y * 20 + x * 20)

    expected_img = 19 * np.ones_like(img)
    expected_img[0, 0] = 0

    background = rolling_ball(img, radius=1)
    assert np.allclose(img - background, expected_img)


@pytest.mark.parametrize("radius", [2, 10, 12.5, 50])
def test_preserve_peaks(radius):
    x, y = np.meshgrid(range(100), range(100))
    img = 0 * x + 0 * y + 10
    img[10, 10] = 20
    img[20, 20] = 35
    img[45, 26] = 156

    expected_img = img - 10
    background = rolling_ball(img, radius)
    assert np.allclose(img - background, expected_img)


@pytest.mark.parametrize("num_threads", [None, 1, 2])
def test_threads(num_threads):
    # not testing if we use multiple threads
    # just checking if the API throws an exception
    img = 23 * np.ones((100, 100), dtype=np.uint8)
    background = rolling_ball(img, 10, num_threads=num_threads)
    background = rolling_ball(img, 10, nansafe=True, num_threads=num_threads)


def test_ndim():
    path = data.image_fetcher.fetch('data/cells.tif')
    image = io.imread(path)[:5, ...]
    rolling_ellipsoid(image, kernel_shape=(3, 100, 100), intensity_vertex=100)
