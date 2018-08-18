# See "Writing benchmarks" in the asv docs for more information.
# http://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np

from skimage import data, img_as_float
from skimage.transform import rescale
from skimage import exposure


class ExposureSuite:
    """Benchmark for exposure routines in scikit-image."""
    def setup(self):
        self.image = img_as_float(data.moon())
        self.image = rescale(self.image, 2.0, anti_aliasing=False)
        # for Contrast stretching
        self.p2, self.p98 = np.percentile(self.image, (2, 98))

    def time_equalize_hist(self):
        result = exposure.equalize_hist(self.image)

    def time_equalize_adapthist(self):
        result = exposure.equalize_adapthist(self.image, clip_limit=0.03)

    def time_rescale_intensity(self):
        result = exposure.rescale_intensity(self.image,
                                            in_range=(self.p2, self.p98))
