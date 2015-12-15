from skimage._shared.kahan import kahan_sum, kahan_cumsum

import numpy as np
from numpy.testing import assert_array_almost_equal


def test_sum():
    rnd = np.random.RandomState(0)
    z = rnd.random_sample((50, 50, 3))

    assert_array_almost_equal(kahan_sum(z), np.sum(z))
    assert_array_almost_equal(kahan_sum(z, axis=1),
                              np.sum(z, axis=1))
    assert_array_almost_equal(kahan_sum(z, axis=0),
                              np.sum(z, axis=0))


def test_cumsum():
    rnd = np.random.RandomState(0)
    z = rnd.random_sample((50, 50, 3))

    assert_array_almost_equal(kahan_cumsum(z), np.cumsum(z))
    assert_array_almost_equal(kahan_cumsum(z, axis=1),
                              np.cumsum(z, axis=1))
    assert_array_almost_equal(kahan_cumsum(z, axis=0),
                              np.cumsum(z, axis=0))


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
