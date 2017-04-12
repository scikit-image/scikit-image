from numpy.testing import assert_array_equal, assert_equal
import numpy as np
import unittest
from skimage.io._plugins._histograms import histograms


class TestHistogram(unittest.TestCase):
    def test_basic(self):
        img = np.ones((50, 50, 3), dtype=np.uint8)
        r, g, b, v = histograms(img, 255)

        for band in (r, g, b, v):
            yield assert_equal, band.sum(), 50 * 50

    def test_counts(self):
        channel = np.arange(255).reshape(51, 5)
        img = np.empty((51, 5, 3), dtype='uint8')
        img[:, :, 0] = channel
        img[:, :, 1] = channel
        img[:, :, 2] = channel
        r, g, b, v = histograms(img, 255)
        assert_array_equal(r, g)
        assert_array_equal(r, b)
        assert_array_equal(r, v)
        assert_array_equal(r, np.ones(255))

if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
