import numpy as np
from numpy.testing import assert_equal

from skimage.measure._ssim import ssim, _as_windows
import scipy.optimize as opt

def test_ssim_patch_range():
    N = 51
    X = (np.random.random((N, N)) * 255).astype(np.uint8)
    Y = (np.random.random((N, N)) * 255).astype(np.uint8)

    assert(ssim(X, Y, win_size=N) < 0.1)
    assert_equal(ssim(X, X, win_size=N), 1)

def test_as_windows():
    X = np.arange(100).reshape((10, 10))
    W = _as_windows(X, win_size=7)
    assert_equal(W.shape[:2], (4, 4))

    W = _as_windows(X, win_size=3)
    assert_equal(W[0, 0], [[0, 1, 2],
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

def test_ssim_grad():
    N = 30
    X = np.random.random((N, N))
    Y = np.random.random((N, N))

    def func(Y):
        return ssim(X, Y)

    def grad(Y):
        return ssim(X, Y, gradient=True)[1]

    assert(np.all(opt.check_grad(func, grad, Y) < 0.05))

#    N = 200
#    X = np.random.random((N, N))
#    Y = np.random.random((N, N))

#    assert(np.all(np.abs(ssim(X, Y, gradient=True))[1] < 1e-2))


if __name__ == "__main__":
    np.testing.run_module_suite()
