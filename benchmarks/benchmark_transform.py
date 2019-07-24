# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt
from skimage.transform import generalized_distance_transform


class edt2d:
    """Benchmark for distance transform in scikit-image."""
    timeout = 120.0
    def setup(self):
        self.case = (1+-1*(np.random.randint(50, size=(4096,4096))//48)).astype('float64')

    def time_scipy_2d(self):
        result = distance_transform_edt(self.case)

    def time_skimage_2d(self):
        result = generalized_distance_transform(self.case)**0.5

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

    def peakmem_scipy_2d(self):
        result = distance_transform_edt(self.case)

    def peakmem_skimage_2d(self):
        result = generalized_distance_transform(self.case)**0.5




class edt3d:
    """Benchmark for distance transform in scikit-image."""
    timeout = 120.0
    def setup(self):
        self.case = (1+-1*(np.random.randint(50, size=(512,512,512))//48)).astype('float64')

    def time_scipy_3d(self):
        result = distance_transform_edt(self.case)

    def time_skimage_3d(self):
        result = generalized_distance_transform(self.case)**0.5

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

    def peakmem_scipy_3d(self):
        result = distance_transform_edt(self.case)

    def peakmem_skimage_3d(self):
        result = generalized_distance_transform(self.case)**0.5
