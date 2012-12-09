import numpy as np
from numpy.testing import assert_array_almost_equal as assert_close
import scipy.ndimage
from skimage.feature import peak


def test_trivial_case():
    trivial = np.zeros((25, 25))
    peak_indices = peak.peak_local_max(trivial, min_distance=1, indices=True)
    assert not peak_indices     # inherent boolean-ness of empty list
    peaks = peak.peak_local_max(trivial, min_distance=1, indices=False)
    assert (peaks.astype(np.bool) == trivial).all()


def test_noisy_peaks():
    peak_locations = [(7, 7), (7, 13), (13, 7), (13, 13)]

    # image with noise of amplitude 0.8 and peaks of amplitude 1
    image = 0.8 * np.random.random((20, 20))
    for r, c in peak_locations:
        image[r, c] = 1

    peaks_detected = peak.peak_local_max(image, min_distance=5)

    assert len(peaks_detected) == len(peak_locations)
    for loc in peaks_detected:
        assert tuple(loc) in peak_locations


def test_relative_threshold():
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1, 1] = 10
    image[3, 3] = 20
    peaks = peak.peak_local_max(image, min_distance=1, threshold_rel=0.5)
    assert len(peaks) == 1
    assert_close(peaks, [(3, 3)])


def test_absolute_threshold():
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1, 1] = 10
    image[3, 3] = 20
    peaks = peak.peak_local_max(image, min_distance=1, threshold_abs=10)
    assert len(peaks) == 1
    assert_close(peaks, [(3, 3)])


def test_constant_image():
    image = 128 * np.ones((20, 20), dtype=np.uint8)
    peaks = peak.peak_local_max(image, min_distance=1)
    assert len(peaks) == 0


def test_flat_peak():
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1:3, 1:3] = 10
    peaks = peak.peak_local_max(image, min_distance=1)
    assert len(peaks) == 4


def test_num_peaks():
    image = np.zeros((7, 7), dtype=np.uint8)
    image[1, 1] = 10
    image[1, 3] = 11
    image[1, 5] = 12
    image[3, 5] = 8
    image[5, 3] = 7
    assert len(peak.peak_local_max(image, min_distance=1)) == 5
    peaks_limited = peak.peak_local_max(image, min_distance=1, num_peaks=2)
    assert len(peaks_limited) == 2
    assert (1, 3) in peaks_limited
    assert (1, 5) in peaks_limited
    peaks_limited = peak.peak_local_max(image, min_distance=1, num_peaks=4)
    assert len(peaks_limited) == 4
    assert (1, 3) in peaks_limited
    assert (1, 5) in peaks_limited
    assert (1, 1) in peaks_limited
    assert (3, 5) in peaks_limited


def test_reorder_labels():
    np.random.seed(21)
    image = np.random.uniform(size=(40, 60))
    i, j = np.mgrid[0:40, 0:60]
    labels = 1 + (i >= 20) + (j >= 30) * 2
    labels[labels == 4] = 5
    i, j = np.mgrid[-3:4, -3:4]
    footprint = (i * i + j * j <= 9)
    expected = np.zeros(image.shape, float)
    for imin, imax in ((0, 20), (20, 40)):
        for jmin, jmax in ((0, 30), (30, 60)):
            expected[imin:imax, jmin:jmax] = scipy.ndimage.maximum_filter(
                image[imin:imax, jmin:jmax], footprint=footprint)
    expected = (expected == image)
    result = peak.peak_local_max(image, labels=labels, min_distance=1,
                                 threshold_rel=0, footprint=footprint,
                                 indices=False, exclude_border=False)
    assert (result == expected).all()


def test_indices_with_labels():
    np.random.seed(21)
    image = np.random.uniform(size=(40, 60))
    i, j = np.mgrid[0:40, 0:60]
    labels = 1 + (i >= 20) + (j >= 30) * 2
    i, j = np.mgrid[-3:4, -3:4]
    footprint = (i * i + j * j <= 9)
    expected = np.zeros(image.shape, float)
    for imin, imax in ((0, 20), (20, 40)):
        for jmin, jmax in ((0, 30), (30, 60)):
            expected[imin:imax, jmin:jmax] = scipy.ndimage.maximum_filter(
                image[imin:imax, jmin:jmax], footprint=footprint)
    expected = (expected == image)
    result = peak.peak_local_max(image, labels=labels, min_distance=1,
                                 threshold_rel=0, footprint=footprint,
                                 indices=True, exclude_border=False)
    assert (result == np.transpose(expected.nonzero())).all()


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
