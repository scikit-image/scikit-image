import numpy as np

from skimage import data, filters, measure

try:
    from skimage.measure._regionprops import PROP_VALS
except ImportError:
    PROP_VALS = []


def init_regionprops_data():
    image = filters.gaussian(data.coins().astype(float), sigma=3)
    # increase size to (2048, 2048) by tiling
    image = np.tile(image, (4, 4))
    label_image = measure.label(image > 130, connectivity=image.ndim)
    intensity_image = image
    return label_image, intensity_image


class RegionpropsTableIndividual:
    param_names = ['prop']
    params = sorted(list(PROP_VALS))

    def setup(self, prop):
        self.label_image, self.intensity_image = init_regionprops_data()

    def time_single_region_property(self, prop):
        measure.regionprops_table(
            self.label_image, self.intensity_image, properties=[prop], cache=True
        )

    # omit peakmem tests to save time (memory usage was minimal)


class RegionpropsTableAll:
    param_names = ['cache']
    params = (False, True)

    def setup(self, cache):
        self.label_image, self.intensity_image = init_regionprops_data()

    def time_regionprops_table_all(self, cache):
        measure.regionprops_table(
            self.label_image, self.intensity_image, properties=PROP_VALS, cache=cache
        )

    # omit peakmem tests to save time (memory usage was minimal)


class MomentsSuite:
    params = (
        [(64, 64), (4096, 2048), (32, 32, 32), (256, 256, 192)],
        [np.uint8, np.float32, np.float64],
        [1, 2, 3],
    )
    param_names = ['shape', 'dtype', 'order']

    """Benchmark for filter routines in scikit-image."""

    def setup(self, shape, dtype, *args):
        rng = np.random.default_rng(1234)
        if np.dtype(dtype).kind in 'iu':
            self.image = rng.integers(0, 256, shape, dtype=dtype)
        else:
            self.image = rng.standard_normal(shape, dtype=dtype)

    def time_moments_raw(self, shape, dtype, order):
        measure.moments(self.image)

    def time_moments_central(self, shape, dtype, order):
        measure.moments_central(self.image)

    def peakmem_reference(self, shape, dtype, order):
        pass

    def peakmem_moments_central(self, shape, dtype, order):
        measure.moments_central(self.image)
