# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np

from skimage import data, img_as_float
from skimage.transform import rescale
from skimage import exposure


class ExposureSuite:
    """Benchmark for exposure routines in scikit-image."""
    def setup(self):
        self.image = img_as_float(data.moon())
        self.image = rescale(self.image, 2.0, anti_aliasing=False)

    def time_equalize_hist(self):
        # Running it 10 times to achieve significant performance time.
        for i in range(10):
            result = exposure.equalize_hist(self.image)
