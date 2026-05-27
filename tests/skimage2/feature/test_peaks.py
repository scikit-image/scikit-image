import time
import pytest

import numpy as np
from numpy.testing import assert_equal
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist, minkowski

from _skimage2.feature._peaks import _ensure_spacing
from _skimage2.feature import peak_local_max


@pytest.mark.parametrize("p", [1, 2, np.inf])
@pytest.mark.parametrize("size", [30, 50, None])
def test_ensure_spacing_trivial(p, size):
    rng = np.random.RandomState(2744269591)

    # --- Empty input
    assert_equal(_ensure_spacing([], p_norm=p), [])

    # --- A unique point
    coord = rng.randn(1, 2)
    assert_equal(coord, _ensure_spacing(coord, p_norm=p, min_split_size=size))

    # --- Verified spacing
    coord = rng.randn(100, 2)

    # --- 0 spacing
    assert_equal(
        coord, _ensure_spacing(coord, spacing=0, p_norm=p, min_split_size=size)
    )

    # Spacing is chosen to be half the minimum distance
    spacing = pdist(coord, metric=minkowski, p=p).min() * 0.5

    out = _ensure_spacing(coord, spacing=spacing, p_norm=p, min_split_size=size)

    assert_equal(coord, out)


@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("size", [2, 10, None])
def test_ensure_spacing_nD(ndim, size):
    coord = np.ones((5, ndim))

    expected = np.ones((1, ndim))

    assert_equal(_ensure_spacing(coord, min_split_size=size), expected)


@pytest.mark.parametrize("p", [1, 2, np.inf])
@pytest.mark.parametrize("size", [50, 100, None])
def test_ensure_spacing_batch_processing(p, size):
    rng = np.random.RandomState(307271047)
    coord = rng.randn(100, 2)

    # --- Consider the average distance between the point as spacing
    spacing = np.median(pdist(coord, metric=minkowski, p=p))

    expected = _ensure_spacing(coord, spacing=spacing, p_norm=p)

    assert_equal(
        _ensure_spacing(coord, spacing=spacing, p_norm=p, min_split_size=size), expected
    )


def test_ensure_spacing_max_batch_size():
    """Small batches are slow, large batches -> large allocations -> also slow.

    https://github.com/scikit-image/scikit-image/pull/6035#discussion_r751518691
    """
    rng = np.random.RandomState(4215035982)
    coords = rng.randint(low=0, high=1848, size=(40000, 2))
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
    rng = np.random.RandomState(584676969)
    coord = rng.randn(100, 2)

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

    @pytest.mark.parametrize("use_labels", [False, True])
    @pytest.mark.parametrize("num_peaks", [None, 2, 100])
    @pytest.mark.parametrize("num_peaks_per_label", [None, 1])
    @pytest.mark.parametrize("min_distance", [1, 2])
    def test_output_sorted_by_intensity(
        self, use_labels, num_peaks, num_peaks_per_label, min_distance
    ):
        image = np.zeros((8, 8))
        image[1, 1] = 1.0
        image[2, 4] = 5.0
        image[5, 2] = 3.0
        image[6, 6] = 2.0
        image[3, 7] = 4.0

        labels = ndi.label(image > 0)[0] if use_labels else None

        peaks = peak_local_max(
            image,
            min_distance=min_distance,
            num_peaks=num_peaks,
            labels=labels,
            num_peaks_per_label=num_peaks_per_label,
        )
        intensities = image[tuple(peaks.T)]
        assert np.all(
            intensities[:-1] >= intensities[1:]
        ), f"output not intensity-sorted: {intensities.tolist()}"

    def test_num_peaks_does_not_enforce_spacing_across_labels(self):
        """`min_distance` is enforced within each label region only, never
        across labels -- even when `num_peaks` truncates the output.

        The two brightest peaks sit in different labels closer than
        `min_distance`; both must be kept.
        """
        image = np.zeros((8, 8))
        image[2, 2] = 5.0  # label A
        image[2, 4] = 4.0  # label B, only 2 px from peak A (< min_distance)
        image[6, 6] = 1.0  # label C, far away

        labels = ndi.label(image > 0)[0]

        peaks = peak_local_max(image, min_distance=3, num_peaks=2, labels=labels)

        # Both close-but-differently-labeled peaks survive, in intensity order.
        assert_equal(peaks, [[2, 2], [2, 4]])
