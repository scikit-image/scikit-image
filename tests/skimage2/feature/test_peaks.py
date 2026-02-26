import time
import pytest

import numpy as np
from numpy.testing import assert_equal
from scipy.spatial.distance import pdist, minkowski

from skimage2.feature._peaks import _ensure_spacing
from skimage2.feature import peak_local_max


@pytest.mark.parametrize("p", [1, 2, np.inf])
@pytest.mark.parametrize("size", [30, 50, None])
def test_ensure_spacing_trivial(p, size):
    # --- Empty input
    assert _ensure_spacing([], p_norm=p) == []

    # --- A unique point
    coord = np.random.randn(1, 2)
    assert np.array_equal(coord, _ensure_spacing(coord, p_norm=p, min_split_size=size))

    # --- Verified spacing
    coord = np.random.randn(100, 2)

    # --- 0 spacing
    assert np.array_equal(
        coord, _ensure_spacing(coord, spacing=0, p_norm=p, min_split_size=size)
    )

    # Spacing is chosen to be half the minimum distance
    spacing = pdist(coord, metric=minkowski, p=p).min() * 0.5

    out = _ensure_spacing(coord, spacing=spacing, p_norm=p, min_split_size=size)

    assert np.array_equal(coord, out)


@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("size", [2, 10, None])
def test_ensure_spacing_nD(ndim, size):
    coord = np.ones((5, ndim))

    expected = np.ones((1, ndim))

    assert np.array_equal(_ensure_spacing(coord, min_split_size=size), expected)


@pytest.mark.parametrize("p", [1, 2, np.inf])
@pytest.mark.parametrize("size", [50, 100, None])
def test_ensure_spacing_batch_processing(p, size):
    coord = np.random.randn(100, 2)

    # --- Consider the average distance between the point as spacing
    spacing = np.median(pdist(coord, metric=minkowski, p=p))

    expected = _ensure_spacing(coord, spacing=spacing, p_norm=p)

    assert np.array_equal(
        _ensure_spacing(coord, spacing=spacing, p_norm=p, min_split_size=size), expected
    )


def test_ensure_spacing_max_batch_size():
    """Small batches are slow, large batches -> large allocations -> also slow.

    https://github.com/scikit-image/scikit-image/pull/6035#discussion_r751518691
    """
    coords = np.random.randint(low=0, high=1848, size=(40000, 2))
    tstart = time.time()
    _ensure_spacing(coords, spacing=100, min_split_size=50, max_split_size=2000)
    dur1 = time.time() - tstart

    tstart = time.time()
    _ensure_spacing(coords, spacing=100, min_split_size=50, max_split_size=20000)
    dur2 = time.time() - tstart

    # Originally checked dur1 < dur2 to assert that the default batch size was
    # faster than a much larger batch size. However, on rare occasion a CI test
    # case would fail with dur1 ~5% larger than dur2. To be more robust to
    # variable load or differences across architectures, we relax this here.
    assert dur1 < 1.33 * dur2


@pytest.mark.parametrize("p", [1, 2, np.inf])
@pytest.mark.parametrize("size", [30, 50, None])
def test_ensure_spacing_p_norm(p, size):
    coord = np.random.randn(100, 2)

    # --- Consider the average distance between the point as spacing
    spacing = np.median(pdist(coord, metric=minkowski, p=p))
    out = _ensure_spacing(coord, spacing=spacing, p_norm=p, min_split_size=size)

    assert pdist(out, metric=minkowski, p=p).min() > spacing


class TestPeakLocalMax:
    def test_p_norm_default(self):
        image = np.zeros((10, 10))
        image[2, 2] = 1
        image[7, 7] = 1

        # With default (p_norm=2, Euclidean distance), peaks are 7.07 apart
        peaks = peak_local_max(image, min_distance=7)
        assert len(peaks) == 2
        peaks = peak_local_max(image, min_distance=8)
        assert len(peaks) == 1

    def test_exclude_border(self):
        image = np.zeros((5, 5, 5))
        image[[1, 0, 0], [0, 1, 0], [0, 0, 1]] = 1
        image[3, 0, 0] = 1
        image[2, 2, 2] = 1

        # exclude_border=0
        result = peak_local_max(image, min_distance=2, exclude_border=0)
        assert_equal(result, [[0, 0, 1], [2, 2, 2], [3, 0, 0]])

        # Default behavior (exclude_border=1) should be the same
        result = peak_local_max(image, min_distance=2)
        assert_equal(result, [[2, 2, 2]])
        result = peak_local_max(image, min_distance=2, exclude_border=1)
        assert_equal(result, [[2, 2, 2]])

    def test_num_peak_float_error(self):
        image = np.zeros((10, 10))
        peak_local_max(image, num_peaks=1.5)
