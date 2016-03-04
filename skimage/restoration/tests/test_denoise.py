import numpy as np
from numpy.testing import run_module_suite, assert_raises, assert_equal

from skimage import restoration, data, color, img_as_float, measure
from skimage._shared._warnings import expected_warnings

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
    denoised_astro = restoration.denoise_tv_chambolle(img, weight=0.1)
    # which dtype?
    assert denoised_astro.dtype in [np.float, np.float32, np.float64]
    from scipy import ndimage as ndi
    grad = ndi.morphological_gradient(img, size=((3, 3)))
    grad_denoised = ndi.morphological_gradient(denoised_astro, size=((3, 3)))
    # test if the total variation has decreased
    assert grad_denoised.dtype == np.float
    assert (np.sqrt((grad_denoised**2).sum())
            < np.sqrt((grad**2).sum()))


def test_denoise_tv_chambolle_multichannel():
    denoised0 = restoration.denoise_tv_chambolle(astro[..., 0], weight=0.1)
    denoised = restoration.denoise_tv_chambolle(astro, weight=0.1,
                                                multichannel=True)
    assert_equal(denoised[..., 0], denoised0)

    # tile astronaut subset to generate 3D+channels data
    astro3 = np.tile(astro[:64, :64, np.newaxis, :], [1, 1, 2, 1])
    # modify along tiled dimension to give non-zero gradient on 3rd axis
    astro3[:, :, 0, :] = 2*astro3[:, :, 0, :]
    denoised0 = restoration.denoise_tv_chambolle(astro3[..., 0], weight=0.1)
    denoised = restoration.denoise_tv_chambolle(astro3, weight=0.1,
                                                multichannel=True)
    assert_equal(denoised[..., 0], denoised0)


def test_denoise_tv_chambolle_float_result_range():
    # astronaut image
    img = astro_gray
    int_astro = np.multiply(img, 255).astype(np.uint8)
    assert np.max(int_astro) > 1
    denoised_int_astro = restoration.denoise_tv_chambolle(int_astro,
                                                          weight=0.1)
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
    res = restoration.denoise_tv_chambolle(mask.astype(np.uint8), weight=0.1)
    assert res.dtype == np.float
    assert res.std() * 255 < mask.std()


def test_denoise_tv_chambolle_1d():
    """Apply the TV denoising algorithm on a 1D sinusoid."""
    x = 125 + 100*np.sin(np.linspace(0, 8*np.pi, 1000))
    x += 20 * np.random.rand(x.size)
    x = np.clip(x, 0, 255)
    res = restoration.denoise_tv_chambolle(x.astype(np.uint8), weight=0.1)
    assert res.dtype == np.float
    assert res.std() * 255 < x.std()


def test_denoise_tv_chambolle_4d():
    """ TV denoising for a 4D input."""
    im = 255 * np.random.rand(8, 8, 8, 8)
    res = restoration.denoise_tv_chambolle(im.astype(np.uint8), weight=0.1)
    assert res.dtype == np.float
    assert res.std() * 255 < im.std()


def test_denoise_tv_chambolle_weighting():
    # make sure a specified weight gives consistent results regardless of
    # the number of input image dimensions
    rstate = np.random.RandomState(1234)
    img2d = astro_gray.copy()
    img2d += 0.15 * rstate.standard_normal(img2d.shape)
    img2d = np.clip(img2d, 0, 1)

    # generate 4D image by tiling
    img4d = np.tile(img2d[..., None, None], (1, 1, 2, 2))

    w = 0.2
    denoised_2d = restoration.denoise_tv_chambolle(img2d, weight=w)
    denoised_4d = restoration.denoise_tv_chambolle(img4d, weight=w)
    assert measure.compare_ssim(denoised_2d,
                                denoised_4d[:, :, 0, 0]) > 0.99


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

    out1 = restoration.denoise_bilateral(img, sigma_color=0.1,
                                         sigma_spatial=20, multichannel=False)
    out2 = restoration.denoise_bilateral(img, sigma_color=0.2,
                                         sigma_spatial=30, multichannel=False)

    # make sure noise is reduced in the checkerboard cells
    assert img[30:45, 5:15].std() > out1[30:45, 5:15].std()
    assert out1[30:45, 5:15].std() > out2[30:45, 5:15].std()


def test_denoise_bilateral_color():
    img = checkerboard.copy()
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)

    out1 = restoration.denoise_bilateral(img, sigma_color=0.1, sigma_spatial=20)
    out2 = restoration.denoise_bilateral(img, sigma_color=0.2, sigma_spatial=30)

    # make sure noise is reduced in the checkerboard cells
    assert img[30:45, 5:15].std() > out1[30:45, 5:15].std()
    assert out1[30:45, 5:15].std() > out2[30:45, 5:15].std()


def test_denoise_bilateral_3d_grayscale():
    img =  np.ones((50, 50, 3))
    assert_raises(ValueError, restoration.denoise_bilateral, img,
                  multichannel=False)


def test_denoise_bilateral_3d_multichannel():
    img = np.ones((50, 50, 50))
    with expected_warnings(["grayscale"]):
        result = restoration.denoise_bilateral(img)

    expected = np.empty_like(img)
    expected.fill(np.nan)

    assert_equal(result, expected)


def test_denoise_bilateral_multidimensional():
    img = np.ones((10, 10, 10, 10))
    assert_raises(ValueError, restoration.denoise_bilateral, img)
    assert_raises(ValueError, restoration.denoise_bilateral, img,
                  multichannel=True)


def test_denoise_bilateral_nan():
    img = np.NaN + np.empty((50, 50))
    out = restoration.denoise_bilateral(img, multichannel=False)
    assert_equal(img, out)

def test_denoise_sigma_range():
    img = checkerboard_gray.copy()
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)
    out1 = restoration.denoise_bilateral(img, sigma_color=0.1,
                                         sigma_spatial=20, multichannel=False)
    with expected_warnings('`sigma_range` has been deprecated in favor of `sigma_color`. '
                           'The `sigma_range` keyword argument will be removed in v0.14'):
        out2 = restoration.denoise_bilateral(img, sigma_range=0.1,
                                             sigma_spatial=20, multichannel=False)
    assert_equal(out1, out2)

def test_denoise_sigma_range_and_sigma_color():
    img = checkerboard_gray.copy()
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)
    out1 = restoration.denoise_bilateral(img, sigma_color=0.1,
                                         sigma_spatial=20, multichannel=False)
    with expected_warnings('`sigma_range` has been deprecated in favor of `sigma_color`. '
                           'The `sigma_range` keyword argument will be removed in v0.14'):
        out2 = restoration.denoise_bilateral(img, sigma_color=0.2, sigma_range=0.1,
                                             sigma_spatial=20, multichannel=False)
    assert_equal(out1, out2)

def test_nl_means_denoising_2d():
    img = np.zeros((40, 40))
    img[10:-10, 10:-10] = 1.
    img += 0.3*np.random.randn(*img.shape)
    denoised = restoration.denoise_nl_means(img, 7, 5, 0.2, fast_mode=True)
    # make sure noise is reduced
    assert img.std() > denoised.std()
    denoised = restoration.denoise_nl_means(img, 7, 5, 0.2, fast_mode=False)
    # make sure noise is reduced
    assert img.std() > denoised.std()


def test_denoise_nl_means_2drgb():
    # reduce image size because nl means is very slow
    img = np.copy(astro[:50, :50])
    # add some random noise
    img += 0.5 * img.std() * np.random.random(img.shape)
    img = np.clip(img, 0, 1)
    denoised = restoration.denoise_nl_means(img, 7, 9, 0.3, fast_mode=True)
    # make sure noise is reduced
    assert img.std() > denoised.std()
    denoised = restoration.denoise_nl_means(img, 7, 9, 0.3, fast_mode=False)
    # make sure noise is reduced
    assert img.std() > denoised.std()


def test_denoise_nl_means_3d():
    img = np.zeros((20, 20, 10))
    img[5:-5, 5:-5, 3:-3] = 1.
    img += 0.3*np.random.randn(*img.shape)
    denoised = restoration.denoise_nl_means(img, 5, 4, 0.2, fast_mode=True,
                                              multichannel=False)
    # make sure noise is reduced
    assert img.std() > denoised.std()
    denoised = restoration.denoise_nl_means(img, 5, 4, 0.2, fast_mode=False,
                                              multichannel=False)
    # make sure noise is reduced
    assert img.std() > denoised.std()


def test_denoise_nl_means_multichannel():
    img = np.zeros((21, 20, 10))
    img[10, 9:11, 2:-2] = 1.
    img += 0.3*np.random.randn(*img.shape)
    denoised_wrong_multichannel = restoration.denoise_nl_means(img,
                    5, 4, 0.1, fast_mode=True, multichannel=True)
    denoised_ok_multichannel = restoration.denoise_nl_means(img,
                    5, 4, 0.1, fast_mode=True, multichannel=False)
    snr_wrong = 10 * np.log10(1. /
                            ((denoised_wrong_multichannel - img)**2).mean())
    snr_ok = 10 * np.log10(1. /
                            ((denoised_ok_multichannel - img)**2).mean())
    assert snr_ok > snr_wrong


def test_denoise_nl_means_wrong_dimension():
    img = np.zeros((5, 5, 5, 5))
    assert_raises(NotImplementedError, restoration.denoise_nl_means, img)


def test_no_denoising_for_small_h():
    img = np.zeros((40, 40))
    img[10:-10, 10:-10] = 1.
    img += 0.3*np.random.randn(*img.shape)
    # very small h should result in no averaging with other patches
    denoised = restoration.denoise_nl_means(img, 7, 5, 0.01, fast_mode=True)
    assert np.allclose(denoised, img)
    denoised = restoration.denoise_nl_means(img, 7, 5, 0.01, fast_mode=False)
    assert np.allclose(denoised, img)


if __name__ == "__main__":
    run_module_suite()
