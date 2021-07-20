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


# For binary morphology all functions ultimately are based on a single erosion
# function in the scipy.ndimage C code, so only benchmark binary_erosion here.

class BinaryMorphology2D(object):

    # skip rectangle as roughly equivalent to square
    param_names = ["shape", "footprint", "radius", "decomposition"]
    params = [
        ((512, 512),),
        ("square", "diamond", "octagon", "disk", "ellipse", "star"),
        (1, 3, 5, 15, 25, 40),
        (None, "sequence", "separable"),
    ]

    def setup(self, shape, footprint, radius, decomposition):
        rng = np.random.default_rng(123)
        # Make an image that is mostly True, with random isolated False areas
        # (so it will not become fully False for any of the footprints).
        self.image = rng.standard_normal(shape) < 3.5
        fp_func = getattr(morphology, footprint)
        allow_decomp = ("rectangle", "square", "diamond", "octagon")
        allow_separable = ("rectangle", "square")
        footprint_kwargs = {}
        if decomposition is not None and footprint not in allow_decomp:
            raise NotImplementedError("decomposition unimplemented")
        elif decomposition == "separable" and footprint not in allow_separable:
            raise NotImplementedError("separable decomposition unavailable")
        if footprint in allow_decomp:
            footprint_kwargs["decomposition"] = decomposition
        if footprint in ["rectangle", "square"]:
            size = 2 * radius + 1
            self.footprint = fp_func(size, **footprint_kwargs)
        elif footprint in ["diamond", "disk"]:
            self.footprint = fp_func(radius, **footprint_kwargs)
        elif footprint == "star":
            # set a so bounding box size is approximately 2*radius + 1
            # size will be 2*a + 1 + 2*floor(a / 2)
            a = max((2 * radius) // 3, 1)
            self.footprint = fp_func(a, **footprint_kwargs)
        elif footprint == "octagon":
            # overall size is m + 2 * n
            # so choose m = n so that overall size is ~ 2*radius + 1
            m = n = max((2 * radius) // 3, 1)
            self.footprint = fp_func(m, n, **footprint_kwargs)
        elif footprint == "ellipse":
            self.footprint = fp_func(radius, radius, **footprint_kwargs)

    def time_erosion(
        self, shape, footprint, radius, *args
    ):
        morphology.binary_erosion(self.image, self.footprint)


class BinaryMorphology3D(object):

    # skip rectangle as roughly equivalent to square
    param_names = ["shape", "footprint", "radius", "decomposition"]
    params = [
        ((128, 128, 128),),
        ("ball", "cube", "octahedron"),
        (1, 3, 5, 10),
        (None, "sequence", "separable"),
    ]

    def setup(self, shape, footprint, radius, decomposition):
        rng = np.random.default_rng(123)
        # make an image that is mostly True, with a few isolated False areas
        self.image = rng.standard_normal(shape) > -3
        fp_func = getattr(morphology, footprint)
        allow_decomp = ("cube", "octahedron")
        allow_separable = ("cube",)
        if decomposition == "separable" and footprint != "cube":
            raise NotImplementedError("separable unavailable")
        footprint_kwargs = {}
        if decomposition is not None and footprint not in allow_decomp:
            raise NotImplementedError("decomposition unimplemented")
        elif decomposition == "separable" and footprint not in allow_separable:
            raise NotImplementedError("separable decomposition unavailable")
        if footprint in allow_decomp:
            footprint_kwargs["decomposition"] = decomposition
        if footprint == "cube":
            size = 2 * radius + 1
            self.footprint = fp_func(size, **footprint_kwargs)
        elif footprint in ["ball", "octahedron"]:
            self.footprint = fp_func(radius, **footprint_kwargs)

    def time_erosion(
        self, shape, footprint, radius, *args
    ):
        morphology.binary_erosion(self.image, self.footprint)


# Repeat the same footprint tests for grayscale morphology
# just need to call morphology.erosion instead of morphology.binary_erosion

class GrayMorphology2D(BinaryMorphology2D):

    def time_erosion(
        self, shape, footprint, radius, *args
    ):
        morphology.erosion(self.image, self.footprint)


class GrayMorphology3D(BinaryMorphology3D):

    def time_erosion(
        self, shape, footprint, radius, *args
    ):
        morphology.erosion(self.image, self.footprint)
