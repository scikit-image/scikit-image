"""Benchmarks for `skimage.morphology`."""

import numpy as np
from numpy.lib import NumpyVersion as Version

import skimage
from skimage import data, filters, morphology, util


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


class Skeletonize3d(object):

    def setup(self, *args):
        try:
            # use a separate skeletonize_3d function on older scikit-image
            if Version(skimage.__version__) < Version('0.16.0'):
                self.skeletonize = morphology.skeletonize_3d
            else:
                self.skeletonize = morphology.skeletonize
        except AttributeError:
            raise NotImplementedError("3d skeletonize unavailable")

        # we stack the horse data 5 times to get an example volume
        self.image = np.stack(5 * [util.invert(data.horse())])

    def time_skeletonize_3d(self):
        self.skeletonize(self.image)

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
        self.skeletonize(self.image)


class RollingBall(object):
    """Benchmark for Rolling Ball algorithm in scikit-image."""

    timeout = 120

    def time_rollingball(self, radius):
        morphology.rolling_ball(data.coins(), radius=radius)
    time_rollingball.params = [25, 50, 75, 100, 150, 200]
    time_rollingball.param_names = ["radius"]

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

    def peakmem_rollingball(self, radius):
        morphology.rolling_ball(data.coins(), radius=radius)
    peakmem_rollingball.params = [25, 50, 75, 100, 150, 200]
    peakmem_rollingball.param_names = ["radius"]

    def time_rollingball_nan(self, radius):
        image = data.coins().astype(np.float_)
        pos = np.arange(np.min(image.shape))
        image[pos, pos] = np.NaN
        morphology.rolling_ball(image, radius=radius, has_nan=True)
    time_rollingball_nan.params = [25, 50, 75, 100, 150, 200]
    time_rollingball_nan.param_names = ["radius"]

    def time_rollingball_ndim(self):
        # would be nice to use cells()
        # how can I load it here from skimage.data?
        image = np.stack([data.coins()] * 20)
        morphology.rolling_ball(image, radius=100)

    def time_rollingball_threads(self, threads):
        morphology.rolling_ball(data.coins(), radius=100, num_threads=threads)
    time_rollingball_threads.params = range(1, 32)
    time_rollingball_threads.param_names = ["threads"]
