import numpy as np
from numpy.testing import *

from skimage.transform import integral_image, integrate

x = (np.random.random((50, 50)) * 255).astype(np.uint8)
s = integral_image(x)


def test_validity():
    y = np.arange(12).reshape((4, 3))

    y = (np.random.random((50, 50)) * 255).astype(np.uint8)
    assert_equal(integral_image(y)[-1, -1],
                 y.sum())


def test_basic():
    assert_equal(x[12:24, 10:20].sum(), integrate(s, 12, 10, 23, 19))
    assert_equal(x[:20, :20].sum(), integrate(s, 0, 0, 19, 19))
    assert_equal(x[:20, 10:20].sum(), integrate(s, 0, 10, 19, 19))
    assert_equal(x[10:20, :20].sum(), integrate(s, 10, 0, 19, 19))


def test_single():
    assert_equal(x[0, 0], integrate(s, 0, 0, 0, 0))
    assert_equal(x[10, 10], integrate(s, 10, 10, 10, 10))


if __name__ == '__main__':
    run_module_suite()
