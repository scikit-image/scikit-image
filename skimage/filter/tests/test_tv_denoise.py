import numpy as np
from numpy.testing import run_module_suite

from skimage import filter, data, color


class TestTvDenoise():

    def test_tv_denoise_2d(self):
        """
        Apply the TV denoising algorithm on the lena image provided
        by scipy
        """
        # lena image
        lena = color.rgb2gray(data.lena())[:256, :256]
        # add noise to lena
        lena += 0.5 * lena.std() * np.random.randn(*lena.shape)
        # clip noise so that it does not exceed allowed range for float images.
        lena = np.clip(lena, 0, 1)
        # denoise
        denoised_lena = filter.tv_denoise(lena, weight=60.0)
        # which dtype?
        assert denoised_lena.dtype in [np.float, np.float32, np.float64]
        from scipy import ndimage
        grad = ndimage.morphological_gradient(lena, size=((3, 3)))
        grad_denoised = ndimage.morphological_gradient(
            denoised_lena, size=((3, 3)))
        # test if the total variation has decreased
        assert grad_denoised.dtype == np.float
        assert (np.sqrt((grad_denoised**2).sum())
                < np.sqrt((grad**2).sum()) / 2)

    def test_tv_denoise_float_result_range(self):
        # lena image
        lena = color.rgb2gray(data.lena())[:256, :256]
        int_lena = np.multiply(lena, 255).astype(np.uint8)
        assert np.max(int_lena) > 1
        denoised_int_lena = filter.tv_denoise(int_lena, weight=60.0)
        # test if the value range of output float data is within [0.0:1.0]
        assert denoised_int_lena.dtype == np.float
        assert np.max(denoised_int_lena) <= 1.0
        assert np.min(denoised_int_lena) >= 0.0

    def test_tv_denoise_3d(self):
        """
        Apply the TV denoising algorithm on a 3D image representing
        a sphere.
        """
        x, y, z = np.ogrid[0:40, 0:40, 0:40]
        mask = (x - 22)**2 + (y - 20)**2 + (z - 17)**2 < 8**2
        mask = 100 * mask.astype(np.float)
        mask += 60
        mask += 20 * np.random.randn(*mask.shape)
        mask[mask < 0] = 0
        mask[mask > 255] = 255
        res = filter.tv_denoise(mask.astype(np.uint8), weight=100)
        assert res.dtype == np.float
        assert res.std() * 255 < mask.std()

        # test wrong number of dimensions
        a = np.random.random((8, 8, 8, 8))
        try:
            res = filter.tv_denoise(a)
        except ValueError:
            pass


if __name__ == "__main__":
    run_module_suite()
