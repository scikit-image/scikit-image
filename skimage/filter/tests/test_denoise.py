import numpy as np
from numpy.testing import run_module_suite, assert_raises

from skimage import filter, data, color, img_as_float


lena = img_as_float(data.lena()[:256, :256])
lena_gray = color.rgb2gray(lena)


def test_denoise_tv_bregman_2d():
    img = lena_gray
    # add some random noise
    img += 0.5 * img.std() * np.random.random(img.shape)
    img = np.clip(img, 0, 1)

    out1 = filter.denoise_tv_bregman(img, weight=10)
    out2 = filter.denoise_tv_bregman(img, weight=5)

    # make sure noise is reduced
    assert img.std() > out1.std()
    assert out1.std() > out2.std()


def test_denoise_tv_bregman_float_result_range():
    # lena image
    img = lena_gray
    int_lena = np.multiply(img, 255).astype(np.uint8)
    assert np.max(int_lena) > 1
    denoised_int_lena = filter.denoise_tv_bregman(int_lena, weight=60.0)
    # test if the value range of output float data is within [0.0:1.0]
    assert denoised_int_lena.dtype == np.float
    assert np.max(denoised_int_lena) <= 1.0
    assert np.min(denoised_int_lena) >= 0.0


def test_denoise_tv_bregman_3d():
    img = lena
    # add some random noise
    img += 0.5 * img.std() * np.random.random(img.shape)
    img = np.clip(img, 0, 1)

    out1 = filter.denoise_tv_bregman(img, weight=10)
    out2 = filter.denoise_tv_bregman(img, weight=5)

    # make sure noise is reduced
    assert img.std() > out1.std()
    assert out1.std() > out2.std()


def test_denoise_bilateral_2d():
    img = lena_gray
    # add some random noise
    img += 0.5 * img.std() * np.random.random(img.shape)
    img = np.clip(img, 0, 1)

    out1 = filter.denoise_bilateral(img, sigma_range=0.1, sigma_spatial=20)
    out2 = filter.denoise_bilateral(img, sigma_range=0.2, sigma_spatial=30)

    # make sure noise is reduced
    assert img.std() > out1.std()
    assert out1.std() > out2.std()


def test_denoise_bilateral_3d():
    img = lena
    # add some random noise
    img += 0.5 * img.std() * np.random.random(img.shape)
    img = np.clip(img, 0, 1)

    out1 = filter.denoise_bilateral(img, sigma_range=0.1, sigma_spatial=20)
    out2 = filter.denoise_bilateral(img, sigma_range=0.2, sigma_spatial=30)

    # make sure noise is reduced
    assert img.std() > out1.std()
    assert out1.std() > out2.std()


if __name__ == "__main__":
    run_module_suite()
