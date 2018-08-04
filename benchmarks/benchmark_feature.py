# See "Writing benchmarks" in the asv docs for more information.
# http://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np
from scipy import ndimage as ndi
from skimage import feature


class FeatureSuite:
    """Benchmark for feature routines in scikit-image."""
    def setup(self):
        self.image = np.zeros((640, 640))
        self.image[320:-320, 320:-320] = 1

        self.image = ndi.rotate(self.image, 15, mode='constant')
        self.image = ndi.gaussian_filter(self.image, 4)
        self.image += 0.2 * np.random.random(self.image.shape)

    def time_canny(self):
        result = feature.canny(self.image)
