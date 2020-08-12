import numpy as np
from skimage import filters

class ThresholdSuite:
    """Benchmark for transform routines in scikit-image."""

    def setup(self):
        self.image = np.zeros((2000, 2000))
        idx = np.arange(500, 1500)
        self.image[idx[::-1], idx] = 255
        self.image[idx, idx] = 255

    def time_sauvola(self):
        result = filters.threshold_sauvola(self.image, window_size=51)
