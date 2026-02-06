import numpy as np

from skimage2.feature import peak_local_max


class TestPeakLocalMax:
    def test_num_peaks(self):
        image = np.zeros((7, 7), dtype=np.uint8)
        image[1, 1] = 10
        image[1, 3] = 11
        image[1, 5] = 12
        image[3, 5] = 8
        image[5, 3] = 7
        assert len(peak_local_max(image, min_distance=1, threshold_abs=0)) == 5
        peaks_limited = peak_local_max(
            image, min_distance=1, threshold_abs=0, num_peaks=2
        )
        assert len(peaks_limited) == 2
        assert (1, 3) in peaks_limited
        assert (1, 5) in peaks_limited
        peaks_limited = peak_local_max(
            image, min_distance=1, threshold_abs=0, num_peaks=4
        )
        assert len(peaks_limited) == 4
        assert (1, 3) in peaks_limited
        assert (1, 5) in peaks_limited
        assert (1, 1) in peaks_limited
        assert (3, 5) in peaks_limited

    def test_p_norm_default(self):
        image = np.zeros((10, 10))
        image[2, 2] = 1
        image[7, 7] = 1

        # With default (p_norm=2, Euclidian distance), peaks are 7.07 apart
        peaks = peak_local_max(image, min_distance=7, exclude_border=0)
        assert len(peaks) == 2
        peaks = peak_local_max(image, min_distance=8, exclude_border=0)
        assert len(peaks) == 1
