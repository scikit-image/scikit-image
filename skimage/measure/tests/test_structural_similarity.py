import numpy as np
from numpy.testing import assert_equal

from skimage.measure import structural_similarity as ssim
import scipy.optimize as opt

def test_ssim_patch_range():
    N = 51
    X = (np.random.random((N, N)) * 255).astype(np.uint8)
    Y = (np.random.random((N, N)) * 255).astype(np.uint8)

    assert(ssim(X, Y, win_size=N) < 0.1)
    assert_equal(ssim(X, X, win_size=N), 1)

def test_ssim_image():
    N = 100
    X = (np.random.random((N, N)) * 255).astype(np.uint8)
    Y = (np.random.random((N, N)) * 255).astype(np.uint8)

    S0 = ssim(X, X, win_size=3)
    assert_equal(S0, 1)

    S1 = ssim(X, Y, win_size=3)
    assert(S1 < 0.3)

## Come up with a better way of testing the gradient
##
## def test_ssim_grad():
##     N = 30
##     X = np.random.random((N, N)) * 255
##     Y = np.random.random((N, N)) * 255

##     def func(Y):
##         return ssim(X, Y, dynamic_range=255)

##     def grad(Y):
##         return ssim(X, Y, dynamic_range=255, gradient=True)[1]

##     assert(np.all(opt.check_grad(func, grad, Y) < 0.05))

def test_ssim_dtype():
    N = 30
    X = np.random.random((N, N))
    Y = np.random.random((N, N))

    S1 = ssim(X, Y)

    X = (X * 255).astype(np.uint8)
    Y = (X * 255).astype(np.uint8)

    S2 = ssim(X, Y)

    assert S1 < 0.1
    assert S2 < 0.1


if __name__ == "__main__":
    np.testing.run_module_suite()
