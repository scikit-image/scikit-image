# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np
from scipy import ndimage as ndi
from skimage import feature, util, img_as_float
from skimage.data import binary_blobs


class FeatureSuite:
    """Benchmark for feature routines in scikit-image."""
    def setup(self):
        self.image = np.zeros((640, 640))
        self.image[320:-320, 320:-320] = 1

        self.image = ndi.rotate(self.image, 15, mode='constant')
        self.image = ndi.gaussian_filter(self.image, 4)
        self.image += 0.2 * np.random.random(self.image.shape)

        self.image_ubyte = util.img_as_ubyte(np.clip(self.image, 0, 1))

    def time_canny(self):
        result = feature.canny(self.image)

    def time_glcm(self):
        pi = np.pi
        result = feature.greycomatrix(self.image_ubyte, distances=[1, 2],
                                      angles=[0, pi/4, pi/2, 3*pi/4])


class RegisterTranslation:
    """Benchmarks for feature.register_translation in scikit-image"""
    param_names = ["ndims", "image_size", "upscale_factor"]
    params = [(2, 3), (32, 100), (1, 5, 10)]

    def setup(self, ndims, image_size, upscale_factor, *args):
        shifts = (-2.3, 1.7, 5.4, -3.2)[:ndims]
        phantom = img_as_float(binary_blobs(length=image_size, n_dim=ndims))
        self.reference_image = np.fft.fftn(phantom)
        self.shifted_image = ndi.fourier_shift(self.reference_image, shifts)

    def time_register_translation(self, ndims, image_size, upscale_factor):
        result = feature.register_translation(self.reference_image,
                                              self.shifted_image,
                                              upscale_factor,
                                              space="fourier")

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

    def peakmem_register_translation(self, ndims, image_size, upscale_factor):
        result = feature.register_translation(self.reference_image,
                                              self.shifted_image,
                                              upscale_factor,
                                              space="fourier")
