import numpy as np
from scipy.signal import get_window as get_window1d
from skimage.filters import get_window
from skimage._shared.testing import parametrize, raises


@parametrize("window",
             [16,
              'hann',
              ('tukey', 0.8)])
@parametrize("size", [5, 8, 15])
@parametrize("ndim", [2, 3, 4])
def test_get_window_shape(window, size, ndim):
    w = get_window(window, size, ndim)
    assert w.ndim == ndim
    assert w.shape[1:] == w.shape[:-1]
    for i in range(1, ndim-1):
        assert np.allclose(w.sum(axis=0), w.sum(axis=i))


@parametrize("window",
             [16,
              'hann',
              ('tukey', 0.8)])
@parametrize("size", [5, 6, 50, 51, 100, 101])
def test_get_window_1d(window, size):
    w = get_window(window, size, ndim=1)
    w1 = get_window1d(window, size, fftbins=False)
    assert np.allclose(w, w1)


def test_get_window_invalid_ndim():
    with raises(ValueError):
        get_window(10, 10, ndim=2.3)
