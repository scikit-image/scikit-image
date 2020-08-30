# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np
from scipy import ndimage as ndi
from skimage import feature, util


class FeatureSuite:
    """Benchmark for feature routines in scikit-image."""
    def setup(self):
        self.image = np.zeros((640, 640))
        self.image[320:-320, 320:-320] = 1

        self.image = ndi.rotate(self.image, 15, mode='constant')
        self.image = ndi.gaussian_filter(self.image, 4)
        self.image += 0.2 * np.random.random(self.image.shape)

        self.image_ubyte = util.img_as_ubyte(np.clip(self.image, 0, 1))

    def time_canny(self):
        result = feature.canny(self.image)

    def time_glcm(self):
        pi = np.pi
        result = feature.greycomatrix(self.image_ubyte, distances=[1, 2],
                                      angles=[0, pi/4, pi/2, 3*pi/4])
