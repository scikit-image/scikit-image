import numpy as np
from numpy.testing import run_module_suite, assert_raises, assert_equal

from skimage import restoration, data, color, img_as_float


np.random.seed(1234)


astro = img_as_float(data.astronaut()[:128, :128])
astro_gray = color.rgb2gray(astro)
checkerboard_gray = img_as_float(data.checkerboard())
checkerboard = color.gray2rgb(checkerboard_gray)


def test_denoise_tv_chambolle_2d():
    # astronaut image
    img = astro_gray.copy()
    # add noise to astronaut
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    # clip noise so that it does not exceed allowed range for float images.
    img = np.clip(img, 0, 1)
    # denoise
    denoised_astro = restoration.denoise_tv_chambolle(img, weight=60.0)
    # which dtype?
    assert denoised_astro.dtype in [np.float, np.float32, np.float64]
    from scipy import ndimage
    grad = ndimage.morphological_gradient(img, size=((3, 3)))
    grad_denoised = ndimage.morphological_gradient(
        denoised_astro, size=((3, 3)))
    # test if the total variation has decreased
    assert grad_denoised.dtype == np.float
    assert (np.sqrt((grad_denoised**2).sum())
            < np.sqrt((grad**2).sum()) / 2)


def test_denoise_tv_chambolle_multichannel():
    denoised0 = restoration.denoise_tv_chambolle(astro[..., 0], weight=60.0)
    denoised = restoration.denoise_tv_chambolle(astro, weight=60.0,
                                                multichannel=True)
    assert_equal(denoised[..., 0], denoised0)


def test_denoise_tv_chambolle_float_result_range():
    # astronaut image
    img = astro_gray
    int_astro = np.multiply(img, 255).astype(np.uint8)
    assert np.max(int_astro) > 1
    denoised_int_astro = restoration.denoise_tv_chambolle(int_astro, weight=60.0)
    # test if the value range of output float data is within [0.0:1.0]
    assert denoised_int_astro.dtype == np.float
    assert np.max(denoised_int_astro) <= 1.0
    assert np.min(denoised_int_astro) >= 0.0


def test_denoise_tv_chambolle_3d():
    """Apply the TV denoising algorithm on a 3D image representing a sphere."""
    x, y, z = np.ogrid[0:40, 0:40, 0:40]
    mask = (x - 22)**2 + (y - 20)**2 + (z - 17)**2 < 8**2
    mask = 100 * mask.astype(np.float)
    mask += 60
    mask += 20 * np.random.rand(*mask.shape)
    mask[mask < 0] = 0
    mask[mask > 255] = 255
    res = restoration.denoise_tv_chambolle(mask.astype(np.uint8), weight=100)
    assert res.dtype == np.float
    assert res.std() * 255 < mask.std()

    # test wrong number of dimensions
    assert_raises(ValueError, restoration.denoise_tv_chambolle,
                  np.random.rand(8, 8, 8, 8))


def test_denoise_tv_bregman_2d():
    img = checkerboard_gray.copy()
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)

    out1 = restoration.denoise_tv_bregman(img, weight=10)
    out2 = restoration.denoise_tv_bregman(img, weight=5)

    # make sure noise is reduced in the checkerboard cells
    assert img[30:45, 5:15].std() > out1[30:45, 5:15].std()
    assert out1[30:45, 5:15].std() > out2[30:45, 5:15].std()


def test_denoise_tv_bregman_float_result_range():
    # astronaut image
    img = astro_gray.copy()
    int_astro = np.multiply(img, 255).astype(np.uint8)
    assert np.max(int_astro) > 1
    denoised_int_astro = restoration.denoise_tv_bregman(int_astro, weight=60.0)
    # test if the value range of output float data is within [0.0:1.0]
    assert denoised_int_astro.dtype == np.float
    assert np.max(denoised_int_astro) <= 1.0
    assert np.min(denoised_int_astro) >= 0.0


def test_denoise_tv_bregman_3d():
    img = checkerboard.copy()
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)

    out1 = restoration.denoise_tv_bregman(img, weight=10)
    out2 = restoration.denoise_tv_bregman(img, weight=5)

    # make sure noise is reduced in the checkerboard cells
    assert img[30:45, 5:15].std() > out1[30:45, 5:15].std()
    assert out1[30:45, 5:15].std() > out2[30:45, 5:15].std()


def test_denoise_bilateral_2d():
    img = checkerboard_gray.copy()
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)

    out1 = restoration.denoise_bilateral(img, sigma_range=0.1,
                                         sigma_spatial=20)
    out2 = restoration.denoise_bilateral(img, sigma_range=0.2,
                                         sigma_spatial=30)

    # make sure noise is reduced in the checkerboard cells
    assert img[30:45, 5:15].std() > out1[30:45, 5:15].std()
    assert out1[30:45, 5:15].std() > out2[30:45, 5:15].std()


def test_denoise_bilateral_3d():
    img = checkerboard.copy()
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)

    out1 = restoration.denoise_bilateral(img, sigma_range=0.1,
                                         sigma_spatial=20)
    out2 = restoration.denoise_bilateral(img, sigma_range=0.2,
                                         sigma_spatial=30)

    # make sure noise is reduced in the checkerboard cells
    assert img[30:45, 5:15].std() > out1[30:45, 5:15].std()
    assert out1[30:45, 5:15].std() > out2[30:45, 5:15].std()


if __name__ == "__main__":
    run_module_suite()
