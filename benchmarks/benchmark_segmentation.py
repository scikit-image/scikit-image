"""Benchmarks for `skimage.segmentation`.

See "Writing benchmarks" in the asv docs for more information.
"""

import numpy as np
from numpy.lib import NumpyVersion as Version

import skimage
from skimage import data, filters, segmentation

from . import _channel_kwarg

try:
    from skimage.segmentation import watershed
except ImportError:
    # older scikit-image had this function under skimage.morphology
    from skimage.morphology import watershed


class SlicSegmentation:
    """Benchmark for segmentation routines in scikit-image."""
    def setup(self):
        self.image = np.random.random((200, 200, 100))
        self.image[:100, :100, :] += 1
        self.image[150:, 150:, :] += 0.5
        self.msk = np.zeros((200, 200, 100))
        self.msk[10:-10, 10:-10, 10:-10] = 1
        self.msk_slice = self.msk[..., 50]
        if Version(skimage.__version__) >= Version('0.17.0'):
            self.slic_kwargs = dict(start_label=1)
        else:
            self.slic_kwargs = {}

    def time_slic_basic(self):
        segmentation.slic(self.image, enforce_connectivity=False,
                          **_channel_kwarg(False), **self.slic_kwargs)

    def time_slic_basic_multichannel(self):
        segmentation.slic(self.image, enforce_connectivity=False,
                          **_channel_kwarg(True), **self.slic_kwargs)

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
        segmentation.slic(self.image, enforce_connectivity=False,
                          **_channel_kwarg(False), **self.slic_kwargs)

    def peakmem_slic_basic_multichannel(self):
        segmentation.slic(self.image, enforce_connectivity=False,
                          **_channel_kwarg(True), **self.slic_kwargs)


class MaskSlicSegmentation(SlicSegmentation):
    """Benchmark for segmentation routines in scikit-image."""
    def setup(self):
        try:
            mask = np.zeros((64, 64)) > 0
            mask[10:-10, 10:-10] = 1
            segmentation.slic(
                np.ones_like(mask), mask=mask, **_channel_kwarg(False)
            )
        except TypeError:
            raise NotImplementedError("masked slic unavailable")

        self.image = np.random.random((200, 200, 100))
        self.image[:100, :100, :] += 1
        self.image[150:, 150:, :] += 0.5
        self.msk = np.zeros((200, 200, 100))
        self.msk[10:-10, 10:-10, 10:-10] = 1
        self.msk_slice = self.msk[..., 50]
        if Version(skimage.__version__) >= Version('0.17.0'):
            self.slic_kwargs = dict(start_label=1)
        else:
            self.slic_kwargs = {}

    def time_mask_slic(self):
        segmentation.slic(self.image, enforce_connectivity=False,
                          mask=self.msk, **_channel_kwarg(False))

    def time_mask_slic_multichannel(self):
        segmentation.slic(self.image, enforce_connectivity=False,
                          mask=self.msk_slice, **_channel_kwarg(True))


class Watershed:

    param_names = ["seed_count", "connectivity", "compactness"]
    params = [(5, 500), (1, 2), (0, 0.01)]

    def setup(self, *args):
        self.image = filters.sobel(data.coins())

    def time_watershed(self, seed_count, connectivity, compactness):
        watershed(self.image, seed_count, connectivity,
                  compactness=compactness)

    def peakmem_reference(self, *args):
        """Provide reference for memory measurement with empty benchmark.

        Peakmem benchmarks measure the maximum amount of RAM used by a
        function. However, this maximum also includes the memory used
        during the setup routine (as of asv 0.2.1; see [1]_).
        Measuring an empty peakmem function might allow us to disambiguate
        between the memory used by setup and the memory used by target (see
        other ``peakmem_`` functions below).

        References
        ----------
        .. [1]: https://asv.readthedocs.io/en/stable/writing_benchmarks.html#peak-memory  # noqa
        """
        pass

    def peakmem_watershed(self, seed_count, connectivity, compactness):
        watershed(self.image, seed_count, connectivity,
                  compactness=compactness)
