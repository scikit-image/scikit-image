# See "Writing benchmarks" in the asv docs for more information.
# http://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np
from skimage import filters


class FiltersSuite:
    """Benchmark for filter routines in scikit-image."""
    def _filt_func(self, r, c):
        return np.exp(-np.hypot(r, c) / 1)

    def setup(self):
        self.image = np.random.random((600, 600))
        self.image[:250, :250] += 1
        self.image[375:, 375] += 0.5
        self.f = filters.LPIFilter2D(self._filt_func)

    def time_sobel(self):
        # Running it 10 times to achieve significant performance time
        for i in range(10):
            result = filters.sobel(self.image)

    def time_inverse(self):
        result = filters.inverse(self.image, predefined_filter=self.f)

    def time_wiener(self):
        result = filters.wiener(self.image, predefined_filter=self.f)
