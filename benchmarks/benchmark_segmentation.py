# See "Writing benchmarks" in the asv docs for more information.
import numpy as np
from skimage import segmentation


class SegmentationSuite:
    """Benchmark for segmentation routines in scikit-image."""
    def setup(self):
        self.image = np.random.random((400, 400, 100))
        self.image[:200, :200, :] += 1
        self.image[300:, 300:, :] += 0.5
        self.msk = np.zeros((400, 400, 100))
        self.msk[10:-10, 10:-10, 10:-10] = 1

    def time_slic_basic(self):
        segmentation.slic(self.image, enforce_connectivity=False)

    def time_mask_slic(self):
        segmentation.slic(self.image, enforce_connectivity=False,
                          mask=self.msk)

    def peakmem_setup(self):
        """peakmem includes the memory used by setup.

        Peakmem benchmarks measure the maximum amount of RAM used by a
        function. However, this maximum also includes the memory used
        by ``setup`` (as of asv 0.2.1; see [1]_)

        Measuring an empty peakmem function might allow us to disambiguate
        between the memory used by setup and the memory used by slic (see
        ``peakmem_slic_basic``, below).

        References
        ----------
        .. [1]: https://asv.readthedocs.io/en/stable/writing_benchmarks.html#peak-memory
        """
        pass

    def peakmem_slic_basic(self):
        segmentation.slic(self.image, enforce_connectivity=False)
