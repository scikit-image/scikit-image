import numpy as np
from numpy.testing import run_module_suite
import scikits.image.filter as F

class TestTvDenoise():

    def test_tv_denoise_2d(self):
        """
        Apply the TV denoising algorithm on the lena image provided
        by scipy
        """
        import scipy
        # lena image
        lena = scipy.lena().astype(np.float)
        # add noise to lena
        lena += 0.5 * lena.std()*np.random.randn(*lena.shape)
        # denoise
        denoised_lena = F.tv_denoise(lena, weight=60.0)
        # which dtype?
        assert denoised_lena.dtype in [np.float, np.float32, np.float64]
        from scipy import ndimage
        grad = ndimage.morphological_gradient(lena, size=((3,3)))
        grad_denoised = ndimage.morphological_gradient(denoised_lena, size=((3,3)))
        # test if the total variation has decreased
        assert np.sqrt((grad_denoised**2).sum()) < np.sqrt((grad**2).sum())
        denoised_lena_int = F.tv_denoise(lena.astype(np.int32), \
                weight=60.0, keep_type=True)
        assert denoised_lena_int.dtype is np.dtype('int32')


    def test_tv_denoise_3d(self):
        """
        Apply the TV denoising algorithm on a 3D image representing
        a sphere.
        """
        x, y, z = np.ogrid[0:40, 0:40, 0:40]
        mask = (x -22)**2 + (y - 20)**2 + (z - 17)**2 < 8**2
        mask = 100 * mask.astype(np.float)
        mask += 60
        mask += 20*np.random.randn(*mask.shape)
        mask[mask < 0] = 0
        mask[mask > 255] = 255
        res = F.tv_denoise(mask.astype(np.uint8), weight=100, keep_type=True)
        assert res.std() < mask.std()
        assert res.dtype is np.dtype('uint8')
        res = F.tv_denoise(mask.astype(np.uint8), weight=100)
        assert res.std() < mask.std()
        assert res.dtype is not np.dtype('uint8')
        # test wrong number of dimensions
        a = np.random.random((8, 8, 8, 8))
        try:
            res = F.tv_denoise(a)
        except ValueError:
            pass


if __name__ == "__main__":
    run_module_suite()
