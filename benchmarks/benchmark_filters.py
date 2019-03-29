# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
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


class ThresholdingSuite:
    """Benchmark for image thresholding routines in scikit-image."""
    def setup(self):
        self.image = np.random.random((4000, 4000))
        self.image[:2000, :2000] += 1
        self.image[3000:, 3000] += 0.5

    def time_threshold_local(self):
        result = filters.threshold_local(self.image, 11)

    def time_threshold_otsu(self):
        result = filters.threshold_otsu(self.image)

    def time_threshold_yen(self):
        result = filters.threshold_yen(self.image)

    def time_threshold_isodata(self):
        result = filters.threshold_isodata(self.image)

    def time_threshold_li(self):
        result = filters.threshold_li(self.image)

    def time_threshold_minimum(self):
        result = filters.threshold_minimum(self.image)

    def time_threshold_mean(self):
        result = filters.threshold_mean(self.image)

    def time_threshold_triangle(self):
        result = filters.threshold_triangle(self.image)

    def time_threshold_niblack(self):
        result = filters.threshold_niblack(self.image)

    def time_threshold_sauvola(self):
        result = filters.threshold_sauvola(self.image)
