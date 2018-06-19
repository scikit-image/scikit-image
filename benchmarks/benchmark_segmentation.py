# See "Writing benchmarks" in the asv docs for more information.
import numpy as np
from skimage import segmentation


class SegmentationSuite:
    """Benchmark for segmentation routines in scikit-image."""
    def setup(self):
        self.image = np.random.random((400, 400, 100))
        self.image[:200, :200, :] += 1
        self.image[300:, 300:, :] += 0.5

    def time_slic_basic(self):
        segmentation.slic(self.image, enforce_connectivity=False)

    def mem_slic_basic(self):
        return segmentation.slic(self.image, enforce_connectivity=False)
