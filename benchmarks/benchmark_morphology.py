"""Benchmarks for `skimage.morphology`."""


from multiprocessing.pool import ThreadPool

import numpy as np

from skimage import data, filters, morphology
from skimage.util import invert


class Watershed(object):

    param_names = ["seed_count", "connectivity", "compactness"]
    params = [(5, 500), (1, 2), (0, 0.01)]

    def setup(self, *args):
        self.image = filters.sobel(data.coins())

    def time_watershed(self, seed_count, connectivity, compactness):
        morphology.watershed(self.image, seed_count, connectivity,
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
        .. [1]: https://asv.readthedocs.io/en/stable/writing_benchmarks.html#peak-memory
        """
        pass

    def peakmem_watershed(self, seed_count, connectivity, compactness):
        morphology.watershed(self.image, seed_count, connectivity,
                             compactness=compactness)


class WatershedParallel(object):

    def setup(self):
        image = filters.sobel(data.coins())
        self.images = ((image, 100) for _ in range(4))

    def time_watershed_parallel(self):
        with ThreadPool(4) as pool:
            pool.starmap(morphology.watershed, self.images)


class Skeletonize3d(object):

    def setup(self, *args):
        # we stack the horse data 5 times to get an example volume
        self.image = np.stack(5 * [invert(data.horse())])

    def time_skeletonize_3d(self):
        morphology.skeletonize_3d(self.image)

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

    def peakmem_skeletonize_3d(self):
        morphology.skeletonize_3d(self.image)
