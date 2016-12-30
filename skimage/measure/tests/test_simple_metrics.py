import numpy as np
from numpy.testing import (run_module_suite, assert_equal, assert_raises,
                           assert_almost_equal)

from skimage.measure import compare_psnr, compare_nrmse, compare_mse
import skimage.data
from skimage._shared._warnings import expected_warnings

np.random.seed(5)
cam = skimage.data.camera()
sigma = 20.0
cam_noisy = np.clip(cam + sigma * np.random.randn(*cam.shape), 0, 255)
cam_noisy = cam_noisy.astype(cam.dtype)


def test_PSNR_vs_IPOL():
    # Tests vs. imdiff result from the following IPOL article and code:
    # http://www.ipol.im/pub/art/2011/g_lmii/
    p_IPOL = 22.4497
    p = compare_psnr(cam, cam_noisy)
    assert_almost_equal(p, p_IPOL, decimal=4)


def test_PSNR_float():
    p_uint8 = compare_psnr(cam, cam_noisy)
    p_float64 = compare_psnr(cam/255., cam_noisy/255., data_range=1)
    assert_almost_equal(p_uint8, p_float64, decimal=5)


def test_PSNR_dynamic_range_and_data_range():
    # Tests deprecation of "dynamic_range" in favor of "data_range"
    out1 = compare_psnr(cam/255., cam_noisy/255., data_range=1)
    with expected_warnings(
            '`dynamic_range` has been deprecated in favor of '
            '`data_range`. The `dynamic_range` keyword argument '
            'will be removed in v0.14'):
        out2 = compare_psnr(cam/255., cam_noisy/255., dynamic_range=1)
    assert_equal(out1, out2)


def test_PSNR_errors():
    assert_raises(ValueError, compare_psnr, cam, cam.astype(np.float32))
    assert_raises(ValueError, compare_psnr, cam, cam[:-1, :])


def test_NRMSE():
    x = np.ones(4)
    y = np.asarray([0., 2., 2., 2.])
    assert_equal(compare_nrmse(y, x, 'mean'), 1/np.mean(y))
    assert_equal(compare_nrmse(y, x, 'Euclidean'), 1/np.sqrt(3))
    assert_equal(compare_nrmse(y, x, 'min-max'), 1/(y.max()-y.min()))


def test_NRMSE_no_int_overflow():
    camf = cam.astype(np.float32)
    cam_noisyf = cam_noisy.astype(np.float32)
    assert_almost_equal(compare_mse(cam, cam_noisy),
                        compare_mse(camf, cam_noisyf))
    assert_almost_equal(compare_nrmse(cam, cam_noisy),
                        compare_nrmse(camf, cam_noisyf))


def test_NRMSE_errors():
    x = np.ones(4)
    assert_raises(ValueError, compare_nrmse,
                  x.astype(np.uint8), x.astype(np.float32))
    assert_raises(ValueError, compare_nrmse, x[:-1], x)
    # invalid normalization name
    assert_raises(ValueError, compare_nrmse, x, x, 'foo')


if __name__ == "__main__":
    run_module_suite()
