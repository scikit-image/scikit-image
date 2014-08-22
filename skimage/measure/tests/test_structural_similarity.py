import numpy as np
from numpy.testing import assert_equal, assert_raises

from skimage.measure import structural_similarity as ssim


np.random.seed(1234)


def test_ssim_patch_range():
    N = 51
    X = (np.random.rand(N, N) * 255).astype(np.uint8)
    Y = (np.random.rand(N, N) * 255).astype(np.uint8)

    assert(ssim(X, Y, win_size=N) < 0.1)
    assert_equal(ssim(X, X, win_size=N), 1)


def test_ssim_image():
    N = 100
    X = (np.random.rand(N, N) * 255).astype(np.uint8)
    Y = (np.random.rand(N, N) * 255).astype(np.uint8)

    S0 = ssim(X, X, win_size=3)
    assert_equal(S0, 1)

    S1 = ssim(X, Y, win_size=3)
    assert(S1 < 0.3)


# NOTE: This test is known to randomly fail on some systems (Mac OS X 10.6)
def test_ssim_grad():
    N = 30
    X = np.random.rand(N, N) * 255
    Y = np.random.rand(N, N) * 255

    f = ssim(X, Y, dynamic_range=255)
    g = ssim(X, Y, dynamic_range=255, gradient=True)

    assert f < 0.05
    assert g[0] < 0.05
    assert np.all(g[1] < 0.05)


def test_ssim_dtype():
    N = 30
    X = np.random.rand(N, N)
    Y = np.random.rand(N, N)

    S1 = ssim(X, Y)

    X = (X * 255).astype(np.uint8)
    Y = (X * 255).astype(np.uint8)

    S2 = ssim(X, Y)

    assert S1 < 0.1
    assert S2 < 0.1


def test_invalid_input():
    X = np.zeros((3, 3), dtype=np.double)
    Y = np.zeros((3, 3), dtype=np.int)
    assert_raises(ValueError, ssim, X, Y)

    Y = np.zeros((4, 4), dtype=np.double)
    assert_raises(ValueError, ssim, X, Y)

    assert_raises(ValueError, ssim, X, X, win_size=8)


if __name__ == "__main__":
    np.testing.run_module_suite()
