import numpy as np
from numpy.testing import assert_equal, assert_raises
from skimage.util import unique_rows


def test_discontiguous_array():
    ar = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], np.uint8)
    ar = ar[::2]
    ar_out = unique_rows(ar)
    desired_ar_out = np.array([[1, 0, 1]], np.uint8)
    assert_equal(ar_out, desired_ar_out)


def test_uint8_array():
    ar = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], np.uint8)
    ar_out = unique_rows(ar)
    desired_ar_out = np.array([[0, 1, 0], [1, 0, 1]], np.uint8)
    assert_equal(ar_out, desired_ar_out)


def test_float_array():
    ar = np.array([[1.1, 0.0, 1.1], [0.0, 1.1, 0.0], [1.1, 0.0, 1.1]],
                  np.float)
    ar_out = unique_rows(ar)
    desired_ar_out = np.array([[0.0, 1.1, 0.0], [1.1, 0.0, 1.1]], np.float)
    assert_equal(ar_out, desired_ar_out)


def test_1d_array():
    ar = np.array([1, 0, 1, 1], np.uint8)
    assert_raises(ValueError, unique_rows, ar)


def test_3d_array():
    ar = np.arange(8).reshape((2, 2, 2))
    assert_raises(ValueError, unique_rows, ar)


if __name__ == '__main__':
    np.testing.run_module_suite()
