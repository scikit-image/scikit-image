# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np

from skimage import data, img_as_float
from skimage import color


class ColorSuite:

    param_names = ['size', 'dtype']
    params = [['small', 'large'], [np.float32, np.float64]]

    """Benchmark for exposure routines in scikit-image."""
    def setup(self, size, dtype):

        self.image_rgb = img_as_float(data.chelsea())
        self.image_hsv = color.rgb2hsv(self.image_rgb)
        self.image_rgb = self.image_rgb.astype(dtype, copy=False)
        self.image_hsv = self.image_hsv.astype(dtype, copy=False)

        # tile the image to get a larger size for benchmarking
        if size == 'large':
            self.image_rgb = np.tile(self.image_rgb, (4, 4, 1))
            self.image_hsv = np.tile(self.image_hsv, (4, 4, 1))

    def time_rgb2hsv(self, size, dtype):
        result = color.rgb2hsv(self.image_rgb)

    def time_hsv2rgb(self, size, dtype):
        result = color.hsv2rgb(self.image_hsv)

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

    def peakmem_rgb2hsv(self, size, dtype):
        result = color.rgb2hsv(self.image_rgb)

    def peakmem_hsv2rgb(self, size, dtype):
        result = color.hsv2rgb(self.image_hsv)
