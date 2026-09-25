import numpy as np
import pytest
from numpy.testing import assert_allclose

from skimage.transform import hough_ridge


def _ring_image(shape=(96, 104), center=(43.4, 57.2), radius=21.5, width=1.2):
    rows, cols = np.indices(shape)
    distance = np.hypot(rows - center[0], cols - center[1])
    return np.exp(-((distance - radius) ** 2) / (2 * width**2))


def test_hough_ridge_subpixel_ring():
    image = _ring_image()
    rings = hough_ridge(
        image,
        (18, 25),
        sigma=1,
        vote_threshold=3,
        circle_threshold=1.5,
        ring_width=2,
    )

    assert rings.shape[1] == 3
    assert_allclose(rings[0], (43.4, 57.2, 21.5), atol=1)


def test_hough_ridge_blank_image():
    rings = hough_ridge(np.zeros((32, 32)), (3, 8))
    assert rings.shape == (0, 3)


def test_hough_ridge_multiple_rings():
    image = _ring_image(shape=(120, 130), center=(35, 38), radius=14)
    image += _ring_image(shape=(120, 130), center=(82, 91), radius=23)
    rings = hough_ridge(
        image,
        (11, 26),
        sigma=1,
        vote_threshold=3,
        circle_threshold=1.5,
        ring_width=2,
    )
    assert_allclose(rings[:2], ((82, 91, 23), (35, 38, 14)), atol=1)


def test_hough_ridge_near_boundary():
    image = _ring_image(shape=(40, 40), center=(7, 8), radius=6)
    rings = hough_ridge(
        image,
        (4, 8),
        sigma=1,
        vote_threshold=2,
        circle_threshold=0.8,
        ring_width=2,
    )
    assert_allclose(rings[0], (7, 8, 6), atol=1)


def test_hough_ridge_large_image_dimension():
    image = _ring_image(shape=(1500, 32), center=(750, 16), radius=8)
    rings = hough_ridge(
        image,
        (6, 10),
        sigma=1,
        vote_threshold=2,
        circle_threshold=1,
        ring_width=2,
    )
    assert_allclose(rings[0], (750, 16, 8), atol=1)


def test_hough_ridge_no_candidate_above_threshold():
    image = _ring_image()
    rings = hough_ridge(image, (18, 25), circle_threshold=10)
    assert rings.shape == (0, 3)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.uint8])
def test_hough_ridge_dtype_and_noncontiguous(dtype):
    image = _ring_image(shape=(80, 84), center=(37, 41), radius=16)
    if np.issubdtype(dtype, np.integer):
        image = (255 * image).astype(dtype)
    else:
        image = image.astype(dtype)
    rings = hough_ridge(
        image[:, ::-1],
        (13, 19),
        sigma=1,
        vote_threshold=3,
        circle_threshold=1.2,
        ring_width=2,
    )
    assert rings.shape[0] >= 1


@pytest.mark.parametrize(
    'image,radii,kwargs,match',
    [
        (np.zeros((3, 3, 3)), (3, 4), {}, 'must be 2D'),
        (np.zeros((0, 3)), (3, 4), {}, 'must not be empty'),
        (np.zeros((10, 10)), (3.0, 4.0), {}, 'two integers'),
        (np.zeros((10, 10)), (2, 4), {}, '3 <= min_radius'),
        (np.zeros((10, 10)), (5, 4), {}, '3 <= min_radius'),
        (np.zeros((10, 10)), (3, 10), {}, 'smaller than'),
        (np.zeros((10, 10)), (3, 4), {'sigma': 0}, 'sigma'),
        (np.zeros((10, 10)), (3, 4), {'vote_threshold': 1.5}, 'vote_threshold'),
        (
            np.zeros((10, 10)),
            (3, 4),
            {'curvature_threshold': np.nan},
            'curvature_threshold',
        ),
    ],
)
def test_hough_ridge_invalid_input(image, radii, kwargs, match):
    with pytest.raises(ValueError, match=match):
        hough_ridge(image, radii, **kwargs)
