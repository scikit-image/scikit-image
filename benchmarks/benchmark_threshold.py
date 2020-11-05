import numpy as np
from skimage.filters.thresholding import threshold_li
from skimage.data import eagle
from skimage.data import camera

class ThresholdingLi:
    """Benchmark for threshold_li in scikit-image."""

    def setup(self):
        self.image = camera()

    def time_integer_image(self):
        result1 = threshold_li(self.image)

    def time_float_image(self):
        result1 = threshold_li(self.image.astype(np.float32))