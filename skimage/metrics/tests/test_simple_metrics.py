from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import assert_equal, assert_almost_equal
from skimage._shared import testing
import numpy as np

from skimage import data
from skimage.metrics import (peak_signal_noise_ratio, normalized_root_mse,
                             mean_squared_error)


np.random.seed(5)
cam = data.camera()
sigma = 20.0
cam_noisy = np.clip(cam + sigma * np.random.randn(*cam.shape), 0, 255)
cam_noisy = cam_noisy.astype(cam.dtype)


def test_PSNR_vs_IPOL():
    # Tests vs. imdiff result from the following IPOL article and code:
    # https://www.ipol.im/pub/art/2011/g_lmii/
    p_IPOL = 22.4497
    p = peak_signal_noise_ratio(cam, cam_noisy)
    assert_almost_equal(p, p_IPOL, decimal=4)


def test_PSNR_float():
    p_uint8 = peak_signal_noise_ratio(cam, cam_noisy)
    p_float64 = peak_signal_noise_ratio(cam / 255., cam_noisy / 255.,
                                        data_range=1)
    assert_almost_equal(p_uint8, p_float64, decimal=5)

    # mixed precision inputs
    p_mixed = peak_signal_noise_ratio(cam / 255., np.float32(cam_noisy / 255.),
                                      data_range=1)
    assert_almost_equal(p_mixed, p_float64, decimal=5)

    # mismatched dtype results in a warning if data_range is unspecified
    with expected_warnings(['Inputs have mismatched dtype']):
        p_mixed = peak_signal_noise_ratio(cam / 255.,
                                          np.float32(cam_noisy / 255.))
    assert_almost_equal(p_mixed, p_float64, decimal=5)


def test_PSNR_errors():
    # shape mismatch
    with testing.raises(ValueError):
        peak_signal_noise_ratio(cam, cam[:-1, :])


def test_NRMSE():
    x = np.ones(4)
    y = np.asarray([0., 2., 2., 2.])
    assert_equal(normalized_root_mse(y, x, normalization='mean'),
                 1 / np.mean(y))
    assert_equal(normalized_root_mse(y, x, normalization='euclidean'),
                 1 / np.sqrt(3))
    assert_equal(normalized_root_mse(y, x, normalization='min-max'),
                 1 / (y.max() - y.min()))

    # mixed precision inputs are allowed
    assert_almost_equal(normalized_root_mse(y, np.float32(x),
                                            normalization='min-max'),
                        1 / (y.max() - y.min()))


def test_NRMSE_no_int_overflow():
    camf = cam.astype(np.float32)
    cam_noisyf = cam_noisy.astype(np.float32)
    assert_almost_equal(mean_squared_error(cam, cam_noisy),
                        mean_squared_error(camf, cam_noisyf))
    assert_almost_equal(normalized_root_mse(cam, cam_noisy),
                        normalized_root_mse(camf, cam_noisyf))


def test_NRMSE_errors():
    x = np.ones(4)
    # shape mismatch
    with testing.raises(ValueError):
        normalized_root_mse(x[:-1], x)
    # invalid normalization name
    with testing.raises(ValueError):
        normalized_root_mse(x, x, normalization='foo')
