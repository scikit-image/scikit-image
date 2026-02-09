import numpy as np
from numpy.testing import assert_equal

from skimage2.feature import peak_local_max


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

        expected_full = np.array([[0, 0, 1], [2, 2, 2], [3, 0, 0]], dtype=int)
        expected_exclude2 = np.array([[2, 2, 2]], dtype=int)

        # exclude_border=0
        result = peak_local_max(image, min_distance=2, exclude_border=0)
        assert_equal(result, expected_full)

        # Default behavior (exclude_border=0) should be the same
        result = peak_local_max(image, min_distance=2)
        assert_equal(result, expected_full)

        # exclude_border=2
        result = peak_local_max(image, min_distance=2, exclude_border=2)
        assert_equal(result, expected_exclude2)

    def test_num_peak_float_error(self):
        image = np.zeros((10, 10))
        peak_local_max(image, num_peaks=1.5)
