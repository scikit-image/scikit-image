import numpy as np
from scipy import ndimage as ndi

from skimage.color import rgb2gray
from skimage import data, img_as_float

# guard against import of a non-existent registration module in older skimage
try:
    from skimage import registration
except ImportError:
    pass

# deal with move and rename of phase_cross_correlation across versions
try:
    from skimage.registration import phase_cross_correlation
except ImportError:
    try:
        from skimage.feature import register_translation

        phase_cross_correlation = register_translation
    except ImportError:
        phase_cross_correlation = None


class RegistrationSuite:
    """Benchmark for registration routines in scikit-image."""

    param_names = ["dtype"]
    params = [(np.float32, np.float64)]

    def setup(self, *args):
        I0, I1, _ = data.stereo_motorcycle()
        self.I0 = rgb2gray(I0)
        self.I1 = rgb2gray(I1)

    def time_tvl1(self, dtype):
        registration.optical_flow_tvl1(self.I0, self.I1, dtype=dtype)

    def time_ilk(self, dtype):
        registration.optical_flow_ilk(self.I0, self.I1, dtype=dtype)


class PhaseCrossCorrelationRegistration:
    """Benchmarks for registration.phase_cross_correlation in scikit-image"""

    param_names = ["ndims", "image_size", "upsample_factor", "dtype"]
    params = [(2, 3), (32, 100), (1, 5, 10), (np.complex64, np.complex128)]

    def setup(self, ndims, image_size, upsample_factor, dtype, *args):
        if phase_cross_correlation is None:
            raise NotImplementedError("phase_cross_correlation unavailable")
        shifts = (-2.3, 1.7, 5.4, -3.2)[:ndims]
        phantom = img_as_float(data.binary_blobs(length=image_size, n_dim=ndims))
        self.reference_image = np.fft.fftn(phantom).astype(dtype, copy=False)
        self.shifted_image = ndi.fourier_shift(self.reference_image, shifts)
        self.shifted_image = self.shifted_image.astype(dtype, copy=False)

    def time_phase_cross_correlation(self, ndims, image_size, upsample_factor, *args):
        phase_cross_correlation(
            self.reference_image,
            self.shifted_image,
            upsample_factor=upsample_factor,
            space="fourier",
        )

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

    def peakmem_phase_cross_correlation(
        self, ndims, image_size, upsample_factor, *args
    ):
        phase_cross_correlation(
            self.reference_image,
            self.shifted_image,
            upsample_factor=upsample_factor,
            space="fourier",
        )
