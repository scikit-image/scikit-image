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
        segmentation.slic(
            self.image,
            enforce_connectivity=False,
            **_channel_kwarg(False),
            **self.slic_kwargs,
        )

    def time_slic_basic_multichannel(self):
        segmentation.slic(
            self.image,
            enforce_connectivity=False,
            **_channel_kwarg(True),
            **self.slic_kwargs,
        )

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
        segmentation.slic(
            self.image,
            enforce_connectivity=False,
            **_channel_kwarg(False),
            **self.slic_kwargs,
        )

    def peakmem_slic_basic_multichannel(self):
        segmentation.slic(
            self.image,
            enforce_connectivity=False,
            **_channel_kwarg(True),
            **self.slic_kwargs,
        )


class MaskSlicSegmentation(SlicSegmentation):
    """Benchmark for segmentation routines in scikit-image."""

    def setup(self):
        try:
            mask = np.zeros((64, 64)) > 0
            mask[10:-10, 10:-10] = 1
            segmentation.slic(np.ones_like(mask), mask=mask, **_channel_kwarg(False))
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
        segmentation.slic(
            self.image,
            enforce_connectivity=False,
            mask=self.msk,
            **_channel_kwarg(False),
        )

    def time_mask_slic_multichannel(self):
        segmentation.slic(
            self.image,
            enforce_connectivity=False,
            mask=self.msk_slice,
            **_channel_kwarg(True),
        )


class Watershed:
    param_names = ["seed_count", "connectivity", "compactness"]
    params = [(5, 500), (1, 2), (0, 0.01)]

    def setup(self, *args):
        self.image = filters.sobel(data.coins())

    def time_watershed(self, seed_count, connectivity, compactness):
        watershed(self.image, seed_count, connectivity, compactness=compactness)

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
        .. [1]: https://asv.readthedocs.io/en/stable/writing_benchmarks.html#peak-memory
        """
        pass

    def peakmem_watershed(self, seed_count, connectivity, compactness):
        watershed(self.image, seed_count, connectivity, compactness=compactness)


class MultiOtsu:
    """Benchmarks for MultiOtsu threshold."""

    param_names = ['classes']
    params = [3, 4, 5]

    def setup(self, *args):
        self.image = data.camera()

    def time_threshold_multiotsu(self, classes):
        segmentation.threshold_multiotsu(self.image, classes=classes)

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
        .. [1]: https://asv.readthedocs.io/en/stable/writing_benchmarks.html#peak-memory
        """
        pass

    def peakmem_threshold_multiotsu(self, classes):
        segmentation.threshold_multiotsu(self.image, classes=classes)


class ThresholdLocalSauvola:
    """Benchmark for transform routines in scikit-image."""

    def setup(self):
        self.image = np.zeros((2000, 2000), dtype=np.uint8)
        self.image3D = np.zeros((30, 300, 300), dtype=np.uint8)

        idx = np.arange(500, 700)
        idx3D = np.arange(10, 200)

        self.image[idx[::-1], idx] = 255
        self.image[idx, idx] = 255

        self.image3D[:, idx3D[::-1], idx3D] = 255
        self.image3D[:, idx3D, idx3D] = 255

    def time_sauvola(self):
        segmentation.threshold_local_sauvola(self.image, window_size=51)

    def time_sauvola_3d(self):
        segmentation.threshold_local_sauvola(self.image3D, window_size=51)


class ThresholdLi:
    """Benchmark for threshold_li in scikit-image."""

    def setup(self):
        try:
            self.image = data.eagle()
        except ValueError:
            raise NotImplementedError("eagle data unavailable")
        self.image_float32 = self.image.astype(np.float32)

    def time_integer_image(self):
        segmentation.threshold_li(self.image)

    def time_float32_image(self):
        segmentation.threshold_li(self.image_float32)
