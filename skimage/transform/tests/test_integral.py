import numpy as np
from numpy.testing import assert_equal

from skimage.transform import integral_image, integrate

np.random.seed(0)
x = (np.random.rand(50, 50) * 255).astype(np.uint8)
s = integral_image(x)


def test_validity():
    y = np.arange(12).reshape((4, 3))

    y = (np.random.rand(50, 50) * 255).astype(np.uint8)
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

def test_vectorized_integrate():
    r0 = np.array([12, 0, 0, 10, 0, 10, 30])
    c0 = np.array([10, 0, 10, 0, 0, 10, 31])
    r1 = np.array([23, 19, 19, 19, 0, 10, 49])
    c1 = np.array([19, 19, 19, 19, 0, 10, 49])

    expected = np.array([x[12:24, 10:20].sum(),
                         x[:20, :20].sum(),
                         x[:20, 10:20].sum(),
                         x[10:20, :20].sum(),
                         x[0,0],
                         x[10, 10],
                         x[30:, 31:].sum()])
    assert_equal(expected, integrate(s, r0, c0, r1, c1))


if __name__ == '__main__':
    from numpy.testing import run_module_suite
    run_module_suite()
