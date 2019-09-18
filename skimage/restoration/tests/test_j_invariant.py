import numpy as np

from skimage._shared.testing import assert_
from skimage.data import binary_blobs
from skimage.data import camera, chelsea
from skimage.measure import compare_mse as mse
from skimage.restoration import (calibrate_denoiser, calibrate_denoiser_search,
                                 invariant_denoise, denoise_wavelet)
from skimage.util import img_as_float, random_noise

test_img = img_as_float(camera())
test_img_color = img_as_float(chelsea())
test_img_3d = img_as_float(binary_blobs(64, n_dim=3)) / 2
noisy_img = random_noise(test_img, mode='gaussian', var=0.01)
noisy_img_color = random_noise(test_img_color, mode='gaussian', var=0.01)
noisy_img_3d = random_noise(test_img_3d, mode='gaussian', var=0.1)


def test_invariant_denoise():
    denoised_img = invariant_denoise(noisy_img, denoise_wavelet)

    denoised_mse = mse(denoised_img, test_img)
    original_mse = mse(noisy_img, test_img)
    assert_(denoised_mse < original_mse)


def test_invariant_denoise_color():
    denoised_img_color = invariant_denoise(noisy_img_color, denoise_wavelet,
                                           multichannel=True)

    denoised_mse = mse(denoised_img_color, test_img_color)
    original_mse = mse(noisy_img_color, test_img_color)
    assert_(denoised_mse < original_mse)


def test_invariant_denoise_3d():
    denoised_img_3d = invariant_denoise(noisy_img_3d, denoise_wavelet)

    denoised_mse = mse(denoised_img_3d, test_img_3d)
    original_mse = mse(noisy_img_3d, test_img_3d)
    assert_(denoised_mse < original_mse)


def test_calibrate_denoiser_search():
    parameter_ranges = {'sigma': np.random.random(5) / 2}
    parameters_tested, losses = calibrate_denoiser_search(
        noisy_img,
        denoise_wavelet,
        denoise_parameters=parameter_ranges
    )

    all_denoised = [invariant_denoise(noisy_img, denoise_wavelet,
                                      denoiser_kwargs=denoiser_kwargs)
                    for denoiser_kwargs in parameters_tested]

    ground_truth_losses = [mse(img, test_img) for img in all_denoised]

    assert_(np.argmin(losses) == np.argmin(ground_truth_losses))


def test_calibrate_denoiser():
    parameter_ranges = {'sigma': np.random.random(5) / 2}

    denoiser = calibrate_denoiser(noisy_img, denoise_wavelet,
                                  denoise_parameters=parameter_ranges)

    denoised_mse = mse(denoiser(noisy_img), test_img)
    original_mse = mse(noisy_img, test_img)
    assert_(denoised_mse < original_mse)


def test_input_image_not_modified():
    input_image = noisy_img.copy()

    parameter_ranges = {'sigma': np.random.random(5) / 2}
    calibrate_denoiser(input_image, denoise_wavelet,
                       denoise_parameters=parameter_ranges)

    assert_(np.all(noisy_img == input_image))


if __name__ == '__main__':
    from numpy import testing

    testing.run_module_suite()
