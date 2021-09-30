import numpy as np

from skimage import data, filters, measure
try:
    from skimage.measure._regionprops import PROP_VALS
except ImportError:
    PROP_VALS = []


def init_regionprops_data():
    image = filters.gaussian(data.coins().astype(float), 3)
    # increase size to (2048, 2048) by tiling
    image = np.tile(image, (4, 4))
    label_image = measure.label(image > 130, connectivity=image.ndim)
    intensity_image = image
    return label_image, intensity_image


class RegionpropsTableIndividual(object):

    param_names = ['prop']
    params = sorted(list(PROP_VALS))

    def setup(self, prop):
        try:
            from skimage.measure import regionprops_table
        except ImportError:
            # regionprops_table was introduced in scikit-image v0.16.0
            raise NotImplementedError("regionprops_table unavailable")
        self.label_image, self.intensity_image = init_regionprops_data()

    def time_single_region_property(self, prop):
        measure.regionprops_table(self.label_image, self.intensity_image,
                                  properties=[prop], cache=True)

    # omit peakmem tests to save time (memory usage was minimal)


class RegionpropsTableAll(object):

    param_names = ['cache']
    params = (False, True)

    def setup(self, cache):
        try:
            from skimage.measure import regionprops_table
        except ImportError:
            # regionprops_table was introduced in scikit-image v0.16.0
            raise NotImplementedError("regionprops_table unavailable")
        self.label_image, self.intensity_image = init_regionprops_data()

    def time_regionprops_table_all(self, cache):
        measure.regionprops_table(self.label_image, self.intensity_image,
                                  properties=PROP_VALS, cache=cache)

    # omit peakmem tests to save time (memory usage was minimal)
