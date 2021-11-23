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


@pytest.mark.parametrize("pad_axes", [None, (0, 1), (0,), (1,)])
@pytest.mark.parametrize("mode", ["full", "same"])
def test_zero_normalized_cross_correlation(pad_axes, mode):
    # Normalized cross_correlation of random arrays
    # should be zero (or about 0)
    np.random.seed(9001)
    x = np.random.random((100, 100))
    np.random.seed(9009)
    y = np.random.random((100, 100))
    cc = cross_correlation(x, y, normalization="zero_normalized", pad_axes=pad_axes, mode=mode)
    if mode is "full":
        assert np.max(cc[50:150, 50:150]) < 0.06
    else:
        assert np.max(cc) < 0.06
    m1 = np.ones(x.shape)
    m2 = np.ones(y.shape)
    cc_masked = cross_correlation(x, y, m1, m2,
                                  normalization="zero_normalized",
                                  pad_axes=pad_axes,
                                  mode=mode,
                                  overlap_ratio=0.0)
    if mode is "full":
        assert np.max(cc_masked[50:150, 50:150]) < 0.06
    else:
        assert np.max(cc_masked) < 0.06
    assert_almost_equal(cc_masked, cc, 4)


@pytest.mark.parametrize("pad_axes", [None, (0, 1), (0,), (1,)])
@pytest.mark.parametrize("mode", ["same", "full"])
def test_cross_correlation(pad_axes, mode):
    # Normalized cross_correlation of random arrays
    # should be zero (or about 0)
    np.random.seed(9001)
    x = np.random.random((100, 100))
    np.random.seed(9009)
    y = np.random.random((100, 100))
    cc = cross_correlation(x, y,
                           normalization=None,
                           pad_axes=pad_axes,
                           mode=mode)
    m1 = np.ones(x.shape)
    m2 = np.ones(y.shape)
    cc_masked = cross_correlation(x, y, m1, m2,
                                  normalization=None,
                                  pad_axes=pad_axes,
                                  mode=mode,
                                  overlap_ratio=0.0)
    assert_almost_equal(cc_masked, cc)

@pytest.mark.parametrize("pad_axes", [None, (0, 1), (0,), (1,)])
@pytest.mark.parametrize("mode", ["same", "full"])
def test_phase_cross_correlation(pad_axes, mode):
    # Normalized cross_correlation of random arrays
    # should be zero (or about 0)
    np.random.seed(9001)
    x = np.random.random((100, 100))
    np.random.seed(9009)
    y = np.random.random((100, 100))
    cc = cross_correlation(x, y,
                           normalization="phase",
                           pad_axes=pad_axes,
                           mode=mode)
    m1 = np.ones(x.shape)
    m2 = np.ones(y.shape)
    cc_masked = cross_correlation(x, y, m1, m2,
                                  normalization="phase",
                                  pad_axes=pad_axes,
                                  mode=mode,
                                  overlap_ratio=0.0)
    print(np.max(cc_masked))
    print(np.max(cc))
    assert_almost_equal(cc_masked, cc)

