import numpy as np

from skimage import feature


def test_noisy_peaks():
    peak_locations = [(7, 7), (7, 13), (13, 7), (13, 13)]

    # image with noise of amplitude 0.8 and peaks of amplitude 1
    image = 0.8 * np.random.random((20, 20))
    for r, c in peak_locations:
        image[r, c] = 1

    peaks_detected = feature.peak_local_max(image, min_distance=5)

    assert len(peaks_detected) == len(peak_locations)
    for loc in peaks_detected:
        assert tuple(loc) in peak_locations


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()

