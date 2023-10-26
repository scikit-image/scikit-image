import inspect

import numpy as np
import scipy.ndimage as ndi

from skimage.data import camera
from skimage import restoration, data, color
from skimage.morphology import binary_dilation

try:
    from skimage.morphology import disk
except ImportError:
    from skimage.morphology import circle as disk
from . import _channel_kwarg, _skip_slow

# inspect signature to automatically handle API changes across versions
if 'num_iter' in inspect.signature(restoration.richardson_lucy).parameters:
    rl_iter_kwarg = dict(num_iter=10)
else:
    rl_iter_kwarg = dict(iterations=10)


class RestorationSuite:
    """Benchmark for restoration routines in scikit image."""

    timeout = 120

    def setup(self):
        nz = 32
        self.volume_f64 = (
            np.stack(
                [
                    camera()[::2, ::2],
                ]
                * nz,
                axis=-1,
            ).astype(float)
            / 255
        )
        self.sigma = 0.05
        self.volume_f64 += self.sigma * np.random.randn(*self.volume_f64.shape)
        self.volume_f32 = self.volume_f64.astype(np.float32)

    def peakmem_setup(self):
        pass

    def time_denoise_nl_means_f64(self):
        restoration.denoise_nl_means(
            self.volume_f64,
            patch_size=3,
            patch_distance=2,
            sigma=self.sigma,
            h=0.7 * self.sigma,
            fast_mode=False,
            **_channel_kwarg(False),
        )

    def time_denoise_nl_means_f32(self):
        restoration.denoise_nl_means(
            self.volume_f32,
            patch_size=3,
            patch_distance=2,
            sigma=self.sigma,
            h=0.7 * self.sigma,
            fast_mode=False,
            **_channel_kwarg(False),
        )

    def time_denoise_nl_means_fast_f64(self):
        restoration.denoise_nl_means(
            self.volume_f64,
            patch_size=3,
            patch_distance=2,
            sigma=self.sigma,
            h=0.7 * self.sigma,
            fast_mode=True,
            **_channel_kwarg(False),
        )

    def time_denoise_nl_means_fast_f32(self):
        restoration.denoise_nl_means(
            self.volume_f32,
            patch_size=3,
            patch_distance=2,
            sigma=self.sigma,
            h=0.7 * self.sigma,
            fast_mode=True,
        )

    def peakmem_denoise_nl_means_f64(self):
        restoration.denoise_nl_means(
            self.volume_f64,
            patch_size=3,
            patch_distance=2,
            sigma=self.sigma,
            h=0.7 * self.sigma,
            fast_mode=False,
            **_channel_kwarg(False),
        )

    def peakmem_denoise_nl_means_f32(self):
        restoration.denoise_nl_means(
            self.volume_f32,
            patch_size=3,
            patch_distance=2,
            sigma=self.sigma,
            h=0.7 * self.sigma,
            fast_mode=False,
        )

    def peakmem_denoise_nl_means_fast_f64(self):
        restoration.denoise_nl_means(
            self.volume_f64,
            patch_size=3,
            patch_distance=2,
            sigma=self.sigma,
            h=0.7 * self.sigma,
            fast_mode=True,
            **_channel_kwarg(False),
        )

    def peakmem_denoise_nl_means_fast_f32(self):
        restoration.denoise_nl_means(
            self.volume_f32,
            patch_size=3,
            patch_distance=2,
            sigma=self.sigma,
            h=0.7 * self.sigma,
            fast_mode=True,
            **_channel_kwarg(False),
        )


class DeconvolutionSuite:
    """Benchmark for restoration routines in scikit image."""

    def setup(self):
        nz = 32
        self.volume_f64 = (
            np.stack(
                [
                    camera()[::2, ::2],
                ]
                * nz,
                axis=-1,
            ).astype(float)
            / 255
        )
        self.sigma = 0.02
        self.psf_f64 = np.ones((5, 5, 5)) / 125
        self.psf_f32 = self.psf_f64.astype(np.float32)
        self.volume_f64 = ndi.convolve(self.volume_f64, self.psf_f64)
        self.volume_f64 += self.sigma * np.random.randn(*self.volume_f64.shape)
        self.volume_f32 = self.volume_f64.astype(np.float32)

    def peakmem_setup(self):
        pass

    def time_richardson_lucy_f64(self):
        restoration.richardson_lucy(self.volume_f64, self.psf_f64, **rl_iter_kwarg)

    def time_richardson_lucy_f32(self):
        restoration.richardson_lucy(self.volume_f32, self.psf_f32, **rl_iter_kwarg)

    # use iterations=1 for peak-memory cases to save time
    def peakmem_richardson_lucy_f64(self):
        restoration.richardson_lucy(self.volume_f64, self.psf_f64, **rl_iter_kwarg)

    def peakmem_richardson_lucy_f32(self):
        restoration.richardson_lucy(self.volume_f32, self.psf_f32, **rl_iter_kwarg)


class RollingBall:
    """Benchmark Rolling Ball algorithm."""

    timeout = 120

    def time_rollingball(self, radius):
        restoration.rolling_ball(data.coins(), radius=radius)

    time_rollingball.params = [25, 50, 100, 200]
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

    peakmem_rollingball.params = [25, 50, 100, 200]
    peakmem_rollingball.param_names = ["radius"]

    def time_rollingball_nan(self, radius):
        image = data.coins().astype(float)
        pos = np.arange(np.min(image.shape))
        image[pos, pos] = np.nan
        restoration.rolling_ball(image, radius=radius, nansafe=True)

    time_rollingball_nan.params = [25, 50, 100, 200]
    time_rollingball_nan.param_names = ["radius"]

    def time_rollingball_ndim(self):
        from skimage.restoration._rolling_ball import ellipsoid_kernel

        image = data.cells3d()[:, 1, ...]
        kernel = ellipsoid_kernel((1, 100, 100), 100)
        restoration.rolling_ball(image, kernel=kernel)

    time_rollingball_ndim.setup = _skip_slow

    def time_rollingball_threads(self, threads):
        restoration.rolling_ball(data.coins(), radius=100, num_threads=threads)

    time_rollingball_threads.params = (0, 2, 4, 8)
    time_rollingball_threads.param_names = ["threads"]


class Inpaint:
    """Benchmark inpainting algorithm."""

    def setup(self):
        image = data.astronaut()

        # Create mask with six block defect regions
        mask = np.zeros(image.shape[:-1], dtype=bool)
        mask[20:60, :20] = 1
        mask[160:180, 70:155] = 1
        mask[30:60, 170:195] = 1
        mask[-60:-30, 170:195] = 1
        mask[-180:-160, 70:155] = 1
        mask[-60:-20, :20] = 1

        # add a few long, narrow defects
        mask[200:205, -200:] = 1
        mask[150:255, 20:23] = 1
        mask[365:368, 60:130] = 1

        # add randomly positioned small point-like defects
        rstate = np.random.RandomState(0)
        for radius in [0, 2, 4]:
            # larger defects are less common
            thresh = 2.75 + 0.25 * radius  # larger defects are less common
            tmp_mask = rstate.randn(*image.shape[:-1]) > thresh
            if radius > 0:
                tmp_mask = binary_dilation(tmp_mask, disk(radius, dtype=bool))
            mask[tmp_mask] = 1

        for layer in range(image.shape[-1]):
            image[np.where(mask)] = 0

        self.image_defect = image
        self.image_defect_gray = color.rgb2gray(image)
        self.mask = mask

    def time_inpaint_rgb(self):
        restoration.inpaint_biharmonic(
            self.image_defect, self.mask, **_channel_kwarg(True)
        )

    def time_inpaint_grey(self):
        restoration.inpaint_biharmonic(
            self.image_defect_gray, self.mask, **_channel_kwarg(False)
        )
