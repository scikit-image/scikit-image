import numpy as np
from numpy.testing import *

from scikits.image.transform import sat, sat_sum

x = (np.random.random((50, 50)) * 255).astype(np.uint8)
s = sat(x)

def test_validity():
    y = np.arange(12).reshape((4,3))

    y = (np.random.random((50, 50)) * 255).astype(np.uint8)
    assert_equal(sat(y)[-1, -1],
                 y.sum())

def test_basic():
    assert_equal(x[12:24, 10:20].sum(), sat_sum(s, 12, 10, 23, 19))
    assert_equal(x[:20, :20].sum(), sat_sum(s, 0, 0, 19, 19))
    assert_equal(x[:20, 10:20].sum(), sat_sum(s, 0, 10, 19, 19))
    assert_equal(x[10:20, :20].sum(), sat_sum(s, 10, 0, 19, 19))

def test_single():
    assert_equal(x[0, 0], sat_sum(s, 0, 0, 0, 0))
    assert_equal(x[10, 10], sat_sum(s, 10, 10, 10, 10))


if __name__ == '__main__':
    run_module_suite()
