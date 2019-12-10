import numpy as np
from scipy.signal import get_window
from skimage.filters import window
from skimage._shared.testing import parametrize, raises


@parametrize("size", [5, 6])
@parametrize("ndim", [2, 3, 4])
def test_window_shape(size, ndim):
    w = window('hann', size, ndim)
    assert w.ndim == ndim
    assert w.shape[1:] == w.shape[:-1]
    for i in range(1, ndim-1):
        assert np.allclose(w.sum(axis=0), w.sum(axis=i))


@parametrize("wintype",
             [16,
              'triang',
              ('tukey', 0.8)])
def test_window_type(wintype):
    w = window(wintype, 9, 2)
    assert w.ndim == 2
    assert w.shape[1:] == w.shape[:-1]
    assert np.allclose(w.sum(axis=0), w.sum(axis=1))


@parametrize("size", [10, 11])
def test_window_1d(size):
    w = window('hann', size, ndim=1)
    w1 = get_window('hann', size, fftbins=False)
    assert np.allclose(w, w1)


def test_window_invalid_ndim():
    with raises(ValueError):
        window(10, 10, ndim=2.3)
    with raises(ValueError):
        window(10, 10, ndim=0)
