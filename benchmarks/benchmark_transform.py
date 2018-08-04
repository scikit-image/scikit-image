# See "Writing benchmarks" in the asv docs for more information.
# http://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np
from scipy import ndimage as ndi
from skimage import transform


class TransformSuite:
    """Benchmark for transform routines in scikit-image."""
    def setup(self):
        self.image = np.zeros((2000, 2000))
        idx = np.arange(500, 1500)
        self.image[idx[::-1], idx] = 255
        self.image[idx, idx] = 255

    def time_hough_line(self):
        # Running it 10 times to achieve significant performance time.
        for i in range(10):
            result1, result2, result3 = transform.hough_line(self.image)
