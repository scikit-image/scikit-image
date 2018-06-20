# See "Writing benchmarks" in the asv docs for more information.
# http://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np
from skimage import filters


class FiltersSuite:
    """Benchmark for filter routines in scikit-image."""
    def setup(self):
        self.image = np.random.random((4000, 4000))
        self.image[:2000, :2000] += 1
        self.image[3000:, 3000] += 0.5

    def time_sobel(self):
        result = filters.sobel(self.image)


