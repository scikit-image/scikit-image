# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np

from skimage import data, exposure, img_as_float32, img_as_float64

image_list = ['moon']
if hasattr(data, 'eagle'):
    # eagle image only available in scikit-image>=0.18
    image_list += ['eagle']


class ExposureSuite:

    param_names = ['image_name', 'dtype']
    params = [image_list, ['uint8', 'float32', 'float64']]

    """Benchmark for exposure routines in scikit-image."""
    def setup(self, image_name, dtype):
        image = getattr(data, image_name)()
        if dtype == 'float32':
            self.image = img_as_float32(image)
        elif dtype == 'float64':
            self.image = img_as_float64(image)
        else:
            self.image = image.astype(dtype, copy=False)
        # for Contrast stretching
        self.p2, self.p98 = np.percentile(self.image, (2, 98))

    def time_equalize_hist(self, *args):
        try:
            # scikit-image <= 0.19.x do not have the method kwarg
            method = 'uint8' if self.image.dtype == np.uint8 else 'float'
            exposure.equalize_hist(self.image, method=method)
        except TypeError:
            exposure.equalize_hist(self.image)

    def time_equalize_adapthist(self, *args):
        exposure.equalize_adapthist(self.image, clip_limit=0.03)

    def time_rescale_intensity(self, *args):
        exposure.rescale_intensity(self.image, in_range=(self.p2, self.p98))

    def time_histogram(self, *args):
        exposure.histogram(self.image)

    def time_gamma_adjust(self, *args):
        exposure.adjust_gamma(self.image)
