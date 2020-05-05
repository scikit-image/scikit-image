from skimage.data import stereo_motorcycle
from skimage.color import rgb2gray
from skimage import registration


class RegistrationSuite(object):
    """Benchmark for registration routines in scikit-image."""
    def setup(self):
        I0, I1, _ = stereo_motorcycle()
        self.I0 = rgb2gray(I0)
        self.I1 = rgb2gray(I1)

    def time_tvl1(self):
        registration.tvl1(self.I0, self.I1)
