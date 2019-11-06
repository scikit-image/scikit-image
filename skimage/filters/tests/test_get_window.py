import numpy as np
from skimage.filters import get_window
from skimage._shared.testing import parametrize


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
