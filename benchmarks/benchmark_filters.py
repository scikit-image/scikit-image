# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np
from skimage import data, filters


class FiltersSuite:
    """Benchmark for filter routines in scikit-image."""
    def setup(self):
        self.image = np.random.random((4000, 4000))
        self.image[:2000, :2000] += 1
        self.image[3000:, 3000] += 0.5

    def time_sobel(self):
        filters.sobel(self.image)


class MultiOtsu(object):
    """Benchmarks for MultiOtsu threshold."""
    param_names = ['classes']
    params = [3, 4, 5]

    def setup(self, *args):
        self.image = data.camera()

    def time_threshold_multiotsu(self, classes):
        filters.threshold_multiotsu(self.image, classes=classes)

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
        filters.threshold_multiotsu(self.image, classes=classes)
