# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np
from skimage import color, data, feature, util


class FeatureSuite:
    """Benchmark for feature routines in scikit-image."""
    def setup(self):
        # Use a real-world image for more realistic features, but tile it to
        # get a larger size for the benchmark.
        self.image = np.tile(color.rgb2gray(data.astronaut()), (4, 4))
        self.image_ubyte = util.img_as_ubyte(self.image)
        self.keypoints = feature.corner_peaks(
                self.image, min_distance=5, threshold_rel=0.1
                )

    def time_canny(self):
        feature.canny(self.image)

    def time_glcm(self):
        pi = np.pi
        feature.greycomatrix(self.image_ubyte, distances=[1, 2],
                             angles=[0, pi/4, pi/2, 3*pi/4])

    def time_brief(self):
        extractor = feature.BRIEF()
        extractor.extract(self.image, self.keypoints)

    def time_hessian_matrix_det(self):
        feature.hessian_matrix_det(self.image, 4)


class HessianSuite:
    """Benchmark for feature routines in scikit-image."""
    param_names = ['ndim']
    params = [2, 3]

    def setup(self, ndim):
        # Use a real-world image for more realistic features, but tile it to
        # get a larger size for the benchmark.
        rng = np.random.default_rng(5)
        if ndim == 2:
            self.image = rng.standard_normal((3840, 2160), dtype=np.float32)
        else:
            self.image = rng.standard_normal((128, 96, 64), dtype=np.float32)
        self.sigma = 3.0
        self.H = feature.hessian_matrix(
            self.image, sigma=self.sigma, use_gaussian_derivatives=False
        )

    def time_hessian_matrix(self, ndim):
        feature.hessian_matrix(
            self.image, self.sigma, use_gaussian_derivatives=False
        )

    def time_hessian_matrix_det(self, ndim):
        feature.hessian_matrix_det(
            self.image, self.sigma, approximate=False
        )

    def time_hessian_matrix_eigvals(self, ndim):
        feature.hessian_matrix_eigvals(self.H)

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
        """  # noqa
        pass

    def peakmem_hessian_matrix(self, ndim):
        feature.hessian_matrix(
            self.image, self.sigma, use_gaussian_derivatives=False
        )

    def peakmem_hessian_matrix_det(self, ndim):
        feature.hessian_matrix_det(self.image, self.sigma, approximate=False)

    def peakmem_hessian_matrix_eigvals(self, ndim):
        feature.hessian_matrix_eigvals(self.H)
