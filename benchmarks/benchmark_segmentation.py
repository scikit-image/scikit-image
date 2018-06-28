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

    def peakmem_setup(self):
        """peakmem includes the memory used by setup.

        This might allow us to disambiguate between the memory used by
        setup and the memory used by slic (see ``peakmem_slic_basic``,
        below).
        """
        pass

    def peakmem_slic_basic(self):
        segmentation.slic(self.image, enforce_connectivity=False)
