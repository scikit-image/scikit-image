import numpy as np
from numpy.testing import assert_equal

from skimage.measure._ssim import ssim, _as_windows

def test_ssim_patch_range():
    N = 51
    X = (np.random.random((N, N)) * 255).astype(np.uint8)
    Y = (np.random.random((N, N)) * 255).astype(np.uint8)

    assert(ssim(X, Y, win_size=N) < 0.1)
    assert_equal(ssim(X, X, win_size=N), 1)

def test_as_windows():
    X = np.arange(100).reshape((10, 10))
    W = _as_windows(X, win_size=7)
    assert_equal(len(W), 16)

    W = _as_windows(X, win_size=3)
    assert_equal(W[0], [[0, 1, 2],
                        [10, 11, 12],
                        [20, 21, 22]])

def test_ssim_image():
    N = 100
    X = (np.random.random((N, N)) * 255).astype(np.uint8)
    Y = (np.random.random((N, N)) * 255).astype(np.uint8)

    S0 = ssim(X, X, win_size=3)
    assert_equal(S0, 1)
    
    S1 = ssim(X, Y, win_size=3)
    assert(S1 < 0.3)

if __name__ == "__main__":
    np.testing.run_module_suite()
