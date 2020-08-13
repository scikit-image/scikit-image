import numpy as np
from skimage import filters

class ThresholdSuite:
    """Benchmark for transform routines in scikit-image."""

    def setup(self):
        self.image = np.zeros((2000, 2000))
        self.image3D = np.zeros((300, 300, 30))

        idx = np.arange(500, 700)
        idx3D = np.arange(100, 100)
        
        self.image[idx[::-1], idx] = 255
        self.image[idx, idx] = 255

        self.image3D[idx3D[::-1], idx3D, idx3D] = 255
        self.image3D[idx3D, idx3D, idx3D] = 255
        
    def time_sauvola(self):
        result = filters.threshold_sauvola(self.image, window_size=51)

    def time_sauvola_3d(self):
        result = filters.threshold_sauvola(self.image3D, window_size=51)
