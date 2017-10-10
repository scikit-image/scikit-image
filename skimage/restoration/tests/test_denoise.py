import numpy as np
from numpy.testing import (run_module_suite, assert_raises, assert_equal,
                           assert_almost_equal, assert_warns, assert_)

from skimage import restoration, data, color, img_as_float, measure
from skimage._shared._warnings import expected_warnings
from skimage.measure import compare_psnr
from skimage.restoration._denoise import _wavelet_threshold

import pywt

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
    assert_(denoised_astro.dtype in [np.float, np.float32, np.float64])
    from scipy import ndimage as ndi
    grad = ndi.morphological_gradient(img, size=((3, 3)))
    grad_denoised = ndi.morphological_gradient(denoised_astro, size=((3, 3)))
    # test if the total variation has decreased
    assert_(grad_denoised.dtype == np.float)
    assert_(np.sqrt((grad_denoised**2).sum()) < np.sqrt((grad**2).sum()))


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
    assert_(np.max(int_astro) > 1)
    denoised_int_astro = restoration.denoise_tv_chambolle(int_astro,
                                                          weight=0.1)
    # test if the value range of output float data is within [0.0:1.0]
    assert_(denoised_int_astro.dtype == np.float)
    assert_(np.max(denoised_int_astro) <= 1.0)
    assert_(np.min(denoised_int_astro) >= 0.0)


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
    assert_(res.dtype == np.float)
    assert_(res.std() * 255 < mask.std())


def test_denoise_tv_chambolle_1d():
    """Apply the TV denoising algorithm on a 1D sinusoid."""
    x = 125 + 100*np.sin(np.linspace(0, 8*np.pi, 1000))
    x += 20 * np.random.rand(x.size)
    x = np.clip(x, 0, 255)
    res = restoration.denoise_tv_chambolle(x.astype(np.uint8), weight=0.1)
    assert_(res.dtype == np.float)
    assert_(res.std() * 255 < x.std())


def test_denoise_tv_chambolle_4d():
    """ TV denoising for a 4D input."""
    im = 255 * np.random.rand(8, 8, 8, 8)
    res = restoration.denoise_tv_chambolle(im.astype(np.uint8), weight=0.1)
    assert_(res.dtype == np.float)
    assert_(res.std() * 255 < im.std())


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
    assert_(measure.compare_ssim(denoised_2d,
                                 denoised_4d[:, :, 0, 0]) > 0.99)


def test_denoise_tv_bregman_2d():
    img = checkerboard_gray.copy()
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)

    out1 = restoration.denoise_tv_bregman(img, weight=10)
    out2 = restoration.denoise_tv_bregman(img, weight=5)

    # make sure noise is reduced in the checkerboard cells
    assert_(img[30:45, 5:15].std() > out1[30:45, 5:15].std())
    assert_(out1[30:45, 5:15].std() > out2[30:45, 5:15].std())


def test_denoise_tv_bregman_float_result_range():
    # astronaut image
    img = astro_gray.copy()
    int_astro = np.multiply(img, 255).astype(np.uint8)
    assert_(np.max(int_astro) > 1)
    denoised_int_astro = restoration.denoise_tv_bregman(int_astro, weight=60.0)
    # test if the value range of output float data is within [0.0:1.0]
    assert_(denoised_int_astro.dtype == np.float)
    assert_(np.max(denoised_int_astro) <= 1.0)
    assert_(np.min(denoised_int_astro) >= 0.0)


def test_denoise_tv_bregman_3d():
    img = checkerboard.copy()
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)

    out1 = restoration.denoise_tv_bregman(img, weight=10)
    out2 = restoration.denoise_tv_bregman(img, weight=5)

    # make sure noise is reduced in the checkerboard cells
    assert_(img[30:45, 5:15].std() > out1[30:45, 5:15].std())
    assert_(out1[30:45, 5:15].std() > out2[30:45, 5:15].std())


def test_denoise_bilateral_2d():
    img = checkerboard_gray.copy()[:50,:50]
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)

    out1 = restoration.denoise_bilateral(img, sigma_color=0.1,
                                         sigma_spatial=10, multichannel=False)
    out2 = restoration.denoise_bilateral(img, sigma_color=0.2,
                                         sigma_spatial=20, multichannel=False)

    # make sure noise is reduced in the checkerboard cells
    assert_(img[30:45, 5:15].std() > out1[30:45, 5:15].std())
    assert_(out1[30:45, 5:15].std() > out2[30:45, 5:15].std())


def test_denoise_bilateral_zeros():
    img = np.zeros((10, 10))
    assert_equal(img, restoration.denoise_bilateral(img, multichannel=False))


def test_denoise_bilateral_constant():
    img = np.ones((10, 10)) * 5
    assert_equal(img, restoration.denoise_bilateral(img, multichannel=False))


def test_denoise_bilateral_color():
    img = checkerboard.copy()[:50, :50]
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)

    out1 = restoration.denoise_bilateral(img, sigma_color=0.1,
                                         sigma_spatial=10, multichannel=True)
    out2 = restoration.denoise_bilateral(img, sigma_color=0.2,
                                         sigma_spatial=20, multichannel=True)

    # make sure noise is reduced in the checkerboard cells
    assert_(img[30:45, 5:15].std() > out1[30:45, 5:15].std())
    assert_(out1[30:45, 5:15].std() > out2[30:45, 5:15].std())


def test_denoise_bilateral_3d_grayscale():
    img = np.ones((50, 50, 3))
    assert_raises(ValueError, restoration.denoise_bilateral, img,
                  multichannel=False)


def test_denoise_bilateral_3d_multichannel():
    img = np.ones((50, 50, 50))
    with expected_warnings(["grayscale"]):
        result = restoration.denoise_bilateral(img, multichannel=True)

    assert_equal(result, img)


def test_denoise_bilateral_multidimensional():
    img = np.ones((10, 10, 10, 10))
    assert_raises(ValueError, restoration.denoise_bilateral, img,
                  multichannel=False)
    assert_raises(ValueError, restoration.denoise_bilateral, img,
                  multichannel=True)


def test_denoise_bilateral_nan():
    img = np.NaN + np.empty((50, 50))
    out = restoration.denoise_bilateral(img, multichannel=False)
    assert_equal(img, out)


def test_denoise_sigma_range():
    img = checkerboard_gray.copy()[:50, :50]
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)
    out1 = restoration.denoise_bilateral(img, sigma_color=0.1,
                                         sigma_spatial=10, multichannel=False)
    with expected_warnings(
            '`sigma_range` has been deprecated in favor of `sigma_color`. '
            'The `sigma_range` keyword argument will be removed in v0.14'):
        out2 = restoration.denoise_bilateral(img, sigma_range=0.1,
                                             sigma_spatial=10,
                                             multichannel=False)
    assert_equal(out1, out2)


def test_denoise_sigma_range_and_sigma_color():
    img = checkerboard_gray.copy()[:50, :50]
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)
    out1 = restoration.denoise_bilateral(img, sigma_color=0.1,
                                         sigma_spatial=10, multichannel=False)
    with expected_warnings(
            '`sigma_range` has been deprecated in favor of `sigma_color`. '
            'The `sigma_range` keyword argument will be removed in v0.14'):
        out2 = restoration.denoise_bilateral(img, sigma_color=0.2,
                                             sigma_range=0.1, sigma_spatial=10,
                                             multichannel=False)
    assert_equal(out1, out2)


def test_nl_means_denoising_2d():
    img = np.zeros((40, 40))
    img[10:-10, 10:-10] = 1.
    img += 0.3*np.random.randn(*img.shape)
    denoised = restoration.denoise_nl_means(img, 7, 5, 0.2, fast_mode=True,
                                            multichannel=True)
    # make sure noise is reduced
    assert_(img.std() > denoised.std())
    denoised = restoration.denoise_nl_means(img, 7, 5, 0.2, fast_mode=False,
                                            multichannel=True)
    # make sure noise is reduced
    assert_(img.std() > denoised.std())


def test_denoise_nl_means_2drgb():
    # reduce image size because nl means is very slow
    img = np.copy(astro[:50, :50])
    # add some random noise
    img += 0.5 * img.std() * np.random.random(img.shape)
    img = np.clip(img, 0, 1)
    denoised = restoration.denoise_nl_means(img, 7, 9, 0.3, fast_mode=True,
                                            multichannel=True)
    # make sure noise is reduced
    assert_(img.std() > denoised.std())
    denoised = restoration.denoise_nl_means(img, 7, 9, 0.3, fast_mode=False,
                                            multichannel=True)
    # make sure noise is reduced
    assert_(img.std() > denoised.std())


def test_denoise_nl_means_3d():
    img = np.zeros((20, 20, 10))
    img[5:-5, 5:-5, 3:-3] = 1.
    img += 0.3*np.random.randn(*img.shape)
    denoised = restoration.denoise_nl_means(img, 5, 4, 0.2, fast_mode=True,
                                            multichannel=False)
    # make sure noise is reduced
    assert_(img.std() > denoised.std())
    denoised = restoration.denoise_nl_means(img, 5, 4, 0.2, fast_mode=False,
                                            multichannel=False)
    # make sure noise is reduced
    assert_(img.std() > denoised.std())


def test_denoise_nl_means_multichannel():
    img = np.zeros((21, 20, 10))
    img[10, 9:11, 2:-2] = 1.
    img += 0.3*np.random.randn(*img.shape)
    denoised_wrong_multichannel = restoration.denoise_nl_means(
        img, 5, 4, 0.1, fast_mode=True, multichannel=True)
    denoised_ok_multichannel = restoration.denoise_nl_means(
        img, 5, 4, 0.1, fast_mode=True, multichannel=False)
    snr_wrong = 10 * np.log10(1. /
                              ((denoised_wrong_multichannel - img)**2).mean())
    snr_ok = 10 * np.log10(1. /
                           ((denoised_ok_multichannel - img)**2).mean())
    assert_(snr_ok > snr_wrong)


def test_denoise_nl_means_wrong_dimension():
    img = np.zeros((5, 5, 5, 5))
    assert_raises(NotImplementedError, restoration.denoise_nl_means, img,
                  multichannel=True)


def test_no_denoising_for_small_h():
    img = np.zeros((40, 40))
    img[10:-10, 10:-10] = 1.
    img += 0.3*np.random.randn(*img.shape)
    # very small h should result in no averaging with other patches
    denoised = restoration.denoise_nl_means(img, 7, 5, 0.01, fast_mode=True,
                                            multichannel=True)
    assert_(np.allclose(denoised, img))
    denoised = restoration.denoise_nl_means(img, 7, 5, 0.01, fast_mode=False,
                                            multichannel=True)
    assert_(np.allclose(denoised, img))


def test_wavelet_denoising():
    rstate = np.random.RandomState(1234)

    # version with one odd-sized dimension
    astro_gray_odd = astro_gray[:, :-1]
    astro_odd = astro[:, :-1]

    for img, multichannel in [(astro_gray, False), (astro_gray_odd, False),
                              (astro_odd, True)]:
        sigma = 0.1
        noisy = img + sigma * rstate.randn(*(img.shape))
        noisy = np.clip(noisy, 0, 1)

        # Verify that SNR is improved when true sigma is used
        denoised = restoration.denoise_wavelet(noisy, sigma=sigma,
                                               multichannel=multichannel)
        psnr_noisy = compare_psnr(img, noisy)
        psnr_denoised = compare_psnr(img, denoised)
        assert_(psnr_denoised > psnr_noisy)

        # Verify that SNR is improved with internally estimated sigma
        denoised = restoration.denoise_wavelet(noisy,
                                               multichannel=multichannel)
        psnr_noisy = compare_psnr(img, noisy)
        psnr_denoised = compare_psnr(img, denoised)
        assert_(psnr_denoised > psnr_noisy)

        # Test changing noise_std (higher threshold, so less energy in signal)
        res1 = restoration.denoise_wavelet(noisy, sigma=2*sigma,
                                           multichannel=multichannel)
        res2 = restoration.denoise_wavelet(noisy, sigma=sigma,
                                           multichannel=multichannel)
        assert_(np.sum(res1**2) <= np.sum(res2**2))


def test_wavelet_threshold():
    rstate = np.random.RandomState(1234)

    img = astro_gray
    sigma = 0.1
    noisy = img + sigma * rstate.randn(*(img.shape))
    noisy = np.clip(noisy, 0, 1)

    # employ a single, uniform threshold instead of BayesShrink sigmas
    denoised = _wavelet_threshold(noisy, wavelet='db1', threshold=sigma)
    psnr_noisy = compare_psnr(img, noisy)
    psnr_denoised = compare_psnr(img, denoised)
    assert_(psnr_denoised > psnr_noisy)


def test_wavelet_denoising_nd():
    rstate = np.random.RandomState(1234)
    for ndim in range(1, 5):
        # Generate a very simple test image
        img = 0.2*np.ones((16, )*ndim)
        img[[slice(5, 13), ] * ndim] = 0.8

        sigma = 0.1
        noisy = img + sigma * rstate.randn(*(img.shape))
        noisy = np.clip(noisy, 0, 1)

        # Verify that SNR is improved with internally estimated sigma
        denoised = restoration.denoise_wavelet(noisy)
        psnr_noisy = compare_psnr(img, noisy)
        psnr_denoised = compare_psnr(img, denoised)
        assert_(psnr_denoised > psnr_noisy)


def test_wavelet_denoising_levels():
    rstate = np.random.RandomState(1234)
    ndim = 2
    N = 256
    wavelet = 'db1'
    # Generate a very simple test image
    img = 0.2*np.ones((N, )*ndim)
    img[[slice(5, 13), ] * ndim] = 0.8

    sigma = 0.1
    noisy = img + sigma * rstate.randn(*(img.shape))
    noisy = np.clip(noisy, 0, 1)

    denoised = restoration.denoise_wavelet(noisy, wavelet=wavelet)
    denoised_1 = restoration.denoise_wavelet(noisy, wavelet=wavelet,
                                             wavelet_levels=1)
    psnr_noisy = compare_psnr(img, noisy)
    psnr_denoised = compare_psnr(img, denoised)
    psnr_denoised_1 = compare_psnr(img, denoised_1)

    # multi-level case should outperform single level case
    assert_(psnr_denoised > psnr_denoised_1 > psnr_noisy)

    # invalid number of wavelet levels results in a ValueError
    max_level = pywt.dwt_max_level(np.min(img.shape),
                                   pywt.Wavelet(wavelet).dec_len)
    assert_raises(ValueError, restoration.denoise_wavelet, noisy,
                  wavelet=wavelet, wavelet_levels=max_level+1)
    assert_raises(ValueError, restoration.denoise_wavelet, noisy,
                  wavelet=wavelet, wavelet_levels=-1)


def test_estimate_sigma_gray():
    rstate = np.random.RandomState(1234)
    # astronaut image
    img = astro_gray.copy()
    sigma = 0.1
    # add noise to astronaut
    img += sigma * rstate.standard_normal(img.shape)

    sigma_est = restoration.estimate_sigma(img, multichannel=False)
    assert_almost_equal(sigma, sigma_est, decimal=2)


def test_estimate_sigma_masked_image():
    # Verify computation on an image with a large, noise-free border.
    # (zero regions will be masked out by _sigma_est_dwt to avoid returning
    #  sigma = 0)
    rstate = np.random.RandomState(1234)
    # uniform image
    img = np.zeros((128, 128))
    center_roi = [slice(32, 96), slice(32, 96)]
    img[center_roi] = 0.8
    sigma = 0.1

    img[center_roi] = sigma * rstate.standard_normal(img[center_roi].shape)

    sigma_est = restoration.estimate_sigma(img, multichannel=False)
    assert_almost_equal(sigma, sigma_est, decimal=1)


def test_estimate_sigma_color():
    rstate = np.random.RandomState(1234)
    # astronaut image
    img = astro.copy()
    sigma = 0.1
    # add noise to astronaut
    img += sigma * rstate.standard_normal(img.shape)

    sigma_est = restoration.estimate_sigma(img, multichannel=True,
                                           average_sigmas=True)
    assert_almost_equal(sigma, sigma_est, decimal=2)

    sigma_list = restoration.estimate_sigma(img, multichannel=True,
                                            average_sigmas=False)
    assert_equal(len(sigma_list), img.shape[-1])
    assert_almost_equal(sigma_list[0], sigma_est, decimal=2)

    # default multichannel=False should raise a warning about last axis size
    assert_warns(UserWarning, restoration.estimate_sigma, img)

def test_wavelet_denoising_args():
    """
    Some of the functions inside wavelet denoising throw an error the wrong
    arguments are passed. This protects against that and verifies that all
    arguments can be passed.
    """
    img = astro
    noisy = img.copy() + 0.1 * np.random.randn(*(img.shape))

    for convert2ycbcr in [True, False]:
        for multichannel in [True, False]:
            for sigma in [0.1, [0.1, 0.1, 0.1], None]:
                if (not multichannel and not convert2ycbcr) or \
                        (isinstance(sigma, list) and not multichannel):
                    continue
                restoration.denoise_wavelet(noisy, sigma=sigma,
                                            convert2ycbcr=convert2ycbcr,
                                            multichannel=multichannel)


def test_multichannel_warnings():
    img = data.astronaut()
    assert_warns(UserWarning, restoration.denoise_bilateral, img)
    assert_warns(UserWarning, restoration.denoise_nl_means, img)


if __name__ == "__main__":
    run_module_suite()
