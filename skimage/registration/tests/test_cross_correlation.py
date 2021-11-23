import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal, assert_array_less, assert_equal)
from scipy.ndimage import fourier_shift, shift as real_shift
import scipy.fft as fft

from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
from skimage.data import camera, brain


from skimage.io import imread

from skimage.registration._cross_correlation import cross_correlation

@pytest.mark.parametrize("upsampling_factor", [1.0, 1.5])
@pytest.mark.parametrize("pad_axes", [None, (0, 1), (0,), (1,)])
@pytest.mark.parametrize("mode", ["full", "same"])
def test_zero_normalized_cross_correlation(pad_axes, mode, upsampling_factor):
    # Normalized cross_correlation of random arrays
    # should be zero (or about 0)
    np.random.seed(9001)
    x = np.random.random((100, 100))
    np.random.seed(9009)
    y = np.random.random((100, 100))
    cc = cross_correlation(x, y, normalization="zero_normalized",
                           pad_axes=pad_axes, mode=mode, upsample_factor=upsampling_factor)
    if mode is "full":
        if upsampling_factor ==1.0:
            assert np.max(cc[50:150, 50:150]) < 0.1
        else:
            assert np.max(cc[75:200, 75:200]) < 0.1
    else:
        print(np.sort(cc))
        assert np.max(cc[1:-1,1:-1]) < 0.1
    m1 = np.ones(x.shape)
    m2 = np.ones(y.shape)
    cc_masked = cross_correlation(x, y, m1, m2,
                                  normalization="zero_normalized",
                                  pad_axes=pad_axes,
                                  mode=mode,
                                  overlap_ratio=0.0, upsample_factor=upsampling_factor)
    if mode is "full":
        if upsampling_factor == 1.0:
            assert np.max(cc[50:150, 50:150]) < 0.1
        else:
            assert np.max(cc[75:100, 75:100]) < 0.1
    else:
        assert np.max(cc_masked) < 0.1
    assert_almost_equal(cc_masked, cc, 2)

@pytest.mark.parametrize("upsampling_factor", [1.0, 1.5])
@pytest.mark.parametrize("pad_axes", [None, (0, 1), (0,), (1,)])
@pytest.mark.parametrize("mode", ["same", "full"])
def test_cross_correlation(pad_axes, mode,upsampling_factor):
    # Normalized cross_correlation of random arrays
    # should be zero (or about 0)
    np.random.seed(9001)
    x = np.random.random((100, 100))
    np.random.seed(9009)
    y = np.random.random((100, 100))
    cc = cross_correlation(x, y,
                           normalization=None,
                           pad_axes=pad_axes,
                           upsample_factor=upsampling_factor,
                           mode=mode)
    m1 = np.ones(x.shape)
    m2 = np.ones(y.shape)
    cc_masked = cross_correlation(x, y, m1, m2,
                                  normalization=None,
                                  pad_axes=pad_axes,
                                  mode=mode,
                                  upsample_factor=upsampling_factor,
                                  overlap_ratio=0.0)
    assert_almost_equal(cc_masked, cc)

@pytest.mark.parametrize("upsampling_factor", [1.0, 1.5])
@pytest.mark.parametrize("pad_axes", [None, (0, 1), (0,), (1,)])
@pytest.mark.parametrize("mode", ["same", "full"])
def test_phase_cross_correlation(pad_axes, mode, upsampling_factor):
    # Normalized cross_correlation of random arrays
    # should be zero (or about 0)
    np.random.seed(9001)
    x = np.random.random((100, 100))
    np.random.seed(9009)
    y = np.random.random((100, 100))
    cc = cross_correlation(x, y,
                           normalization="phase",
                           pad_axes=pad_axes,
                           upsample_factor=upsampling_factor,
                           mode=mode)
    m1 = np.ones(x.shape)
    m2 = np.ones(y.shape)
    cc_masked = cross_correlation(x, y, m1, m2,
                                  normalization="phase",
                                  pad_axes=pad_axes,
                                  upsample_factor=upsampling_factor,
                                  mode=mode,
                                  overlap_ratio=0.0)
    assert_almost_equal(cc_masked, cc)

def test_normmalized_cross_correlation_mask():
    # Normalized cross_correlation of random arrays
    # should be zero (or about 0)
    np.random.seed(9001)
    x = np.random.random((100, 100))
    np.random.seed(9009)
    y = np.random.random((100, 100))
    cc = cross_correlation(x, y,
                           normalization="zero_normalized",
                           pad_axes=None,
                           mode="same")
    m1 = np.ones(x.shape)
    m2 = np.ones(y.shape)
    m1[::10, ::10] = 0
    cc_masked = cross_correlation(x, y, m1, m2,
                                  normalization="zero_normalized",
                                  pad_axes=None,
                                  mode="same",
                                  overlap_ratio=0.1)
    assert_almost_equal(cc_masked, cc,2)

def test_phase_cross_correlation_masked():
    # Normalized cross_correlation of random arrays
    # should be zero (or about 0)
    np.random.seed(9001)
    x = np.random.random((100, 100))
    x[20:25, 20:25]=1
    np.random.seed(9009)
    y = np.random.random((100, 100))
    y[40:45, 30:35]=1
    cc = cross_correlation(x, y,
                           normalization="phase",
                           pad_axes=None,
                           mode="same")
    m1 = np.ones(x.shape)
    m2 = np.ones(y.shape)
    m1[50:80, 50:80] = 0
    m2[60:90, 60:90] = 0
    cc_masked = cross_correlation(x, y, m1, m2,
                                  normalization="phase",
                                  pad_axes=None,
                                  mode="same",
                                  overlap_ratio=0.0)

    assert_almost_equal(cc_masked, cc, 1)
    assert_equal(np.argmax(cc_masked), np.argmax(cc))