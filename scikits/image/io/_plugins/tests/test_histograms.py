from numpy.testing import *
import numpy as np

import scikits.image.io._plugins._colormixer as cm
from scikits.image.io._plugins._histograms import histograms

class TestHistogram:
    def test_basic(self):
        img = np.ones((50, 50, 3), dtype=np.uint8)
        r, g, b, v = histograms(img, 255)

        for band in (r, g, b, v):
            yield assert_equal, band.sum(), 50*50

if __name__ == "__main__":
    run_module_suite()
