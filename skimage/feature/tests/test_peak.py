import numpy as np
from numpy.testing import assert_array_almost_equal as assert_close

from skimage.feature import peak


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
    image[3, 3] = 21
    peaks = peak.peak_local_max(image, min_distance=1, threshold_rel=0.5)
    assert len(peaks) == 1
    assert_close(peaks, [(3, 3)])


def test_absolute_threshold():
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1, 1] = 10
    image[3, 3] = 21
    peaks = peak.peak_local_max(image, min_distance=1, threshold_abs=11)
    assert len(peaks) == 1
    assert_close(peaks, [(3, 3)])

if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()

