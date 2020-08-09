from skimage.color import rgb2gray
from skimage import data

# guard against import of a non-existant registration module in older skimage
try:
    from skimage import registration
except ImportError:
    pass


class RegistrationSuite(object):
    """Benchmark for registration routines in scikit-image."""
    def setup(self):
        try:
            from skimage.registration import optical_flow_tvl1
        except ImportError:
            raise NotImplementedError("optical_flow_tvl1 unavailable")
        I0, I1, _ = data.stereo_motorcycle()
        self.I0 = rgb2gray(I0)
        self.I1 = rgb2gray(I1)

    def time_tvl1(self):
        registration.optical_flow_tvl1(self.I0, self.I1)
