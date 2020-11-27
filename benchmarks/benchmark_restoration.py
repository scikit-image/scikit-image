import numpy as np
from skimage.data import camera
from skimage import restoration, data, io
import scipy.ndimage as ndi


class RestorationSuite:
    """Benchmark for restoration routines in scikit image."""
    def setup(self):
        nz = 32
        self.volume_f64 = np.stack([camera()[::2, ::2], ] * nz,
                                   axis=-1).astype(float) / 255
        self.sigma = .05
        self.volume_f64 += self.sigma * np.random.randn(*self.volume_f64.shape)
        self.volume_f32 = self.volume_f64.astype(np.float32)

    def peakmem_setup(self):
        pass

    def time_denoise_nl_means_f64(self):
        restoration.denoise_nl_means(self.volume_f64, patch_size=3,
                                     patch_distance=2, sigma=self.sigma,
                                     h=0.7 * self.sigma, fast_mode=False,
                                     multichannel=False)

    def time_denoise_nl_means_f32(self):
        restoration.denoise_nl_means(self.volume_f32, patch_size=3,
                                     patch_distance=2, sigma=self.sigma,
                                     h=0.7 * self.sigma, fast_mode=False,
                                     multichannel=False)

    def time_denoise_nl_means_fast_f64(self):
        restoration.denoise_nl_means(self.volume_f64, patch_size=3,
                                     patch_distance=2, sigma=self.sigma,
                                     h=0.7 * self.sigma, fast_mode=True,
                                     multichannel=False)

    def time_denoise_nl_means_fast_f32(self):
        restoration.denoise_nl_means(self.volume_f32, patch_size=3,
                                     patch_distance=2, sigma=self.sigma,
                                     h=0.7 * self.sigma, fast_mode=True)

    def peakmem_denoise_nl_means_f64(self):
        restoration.denoise_nl_means(self.volume_f64, patch_size=3,
                                     patch_distance=2,  sigma=self.sigma,
                                     h=0.7 * self.sigma, fast_mode=False,
                                     multichannel=False)

    def peakmem_denoise_nl_means_f32(self):
        restoration.denoise_nl_means(self.volume_f32, patch_size=3,
                                     patch_distance=2, sigma=self.sigma,
                                     h=0.7 * self.sigma, fast_mode=False)

    def peakmem_denoise_nl_means_fast_f64(self):
        restoration.denoise_nl_means(self.volume_f64, patch_size=3,
                                     patch_distance=2, sigma=self.sigma,
                                     h=0.7 * self.sigma, fast_mode=True,
                                     multichannel=False)

    def peakmem_denoise_nl_means_fast_f32(self):
        restoration.denoise_nl_means(self.volume_f32, patch_size=3,
                                     patch_distance=2, sigma=self.sigma,
                                     h=0.7 * self.sigma, fast_mode=True,
                                     multichannel=False)


class DeconvolutionSuite:
    """Benchmark for restoration routines in scikit image."""
    def setup(self):
        nz = 32
        self.volume_f64 = np.stack([camera()[::2, ::2], ] * nz,
                                   axis=-1).astype(float) / 255
        self.sigma = .02
        self.psf_f64 = np.ones((5, 5, 5)) / 125
        self.psf_f32 = self.psf_f64.astype(np.float32)
        self.volume_f64 = ndi.convolve(self.volume_f64, self.psf_f64)
        self.volume_f64 += self.sigma * np.random.randn(*self.volume_f64.shape)
        self.volume_f32 = self.volume_f64.astype(np.float32)

    def peakmem_setup(self):
        pass

    def time_richardson_lucy_f64(self):
        restoration.richardson_lucy(self.volume_f64, self.psf_f64,
                                    iterations=10)

    def time_richardson_lucy_f32(self):
        restoration.richardson_lucy(self.volume_f32, self.psf_f32,
                                    iterations=10)

    # use iterations=1 for peak-memory cases to save time
    def peakmem_richardson_lucy_f64(self):
        restoration.richardson_lucy(self.volume_f64, self.psf_f64,
                                    iterations=1)

    def peakmem_richardson_lucy_f32(self):
        restoration.richardson_lucy(self.volume_f32, self.psf_f32,
                                    iterations=1)


class RollingBall(object):
    """Benchmark Rolling Ball algorithm."""

    timeout = 120

    def time_rollingball(self, radius):
        restoration.rolling_ball(data.coins(), radius=radius)
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
        restoration.rolling_ball(data.coins(), radius=radius)
    peakmem_rollingball.params = [25, 50, 75, 100, 150, 200]
    peakmem_rollingball.param_names = ["radius"]

    def time_rollingball_nan(self, radius):
        image = data.coins().astype(np.float_)
        pos = np.arange(np.min(image.shape))
        image[pos, pos] = np.NaN
        restoration.rolling_ball(image, radius=radius, nansafe=True)
    time_rollingball_nan.params = [25, 50, 75, 100, 150, 200]
    time_rollingball_nan.param_names = ["radius"]

    def time_rollingball_ndim(self):
        from skimage.restoration.rolling_ball import ellipsoid_kernel
        image = data.cells3d()[:, 1, ...]
        kernel = ellipsoid_kernel((1, 100, 100), 100)
        restoration.rolling_ball(
            image, kernel=kernel)

    def time_rollingball_threads(self, threads):
        restoration.rolling_ball(data.coins(), radius=100, num_threads=threads)
    time_rollingball_threads.params = range(0, 9)
    time_rollingball_threads.param_names = ["threads"]
