import numpy as np
from skimage import transform

class TransformSuite:
    """Benchmark for transform routines in scikit-image."""

    def setup(self):
        self.image = np.zeros((2000, 2000))
        idx = np.arange(500, 1500)
        self.image[idx[::-1], idx] = 255
        self.image[idx, idx] = 255

    def time_hough_line(self):
        result1, result2, result3 = transform.hough_line(self.image)
