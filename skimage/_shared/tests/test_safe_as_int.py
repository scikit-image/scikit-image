import numpy as np
from skimage._shared.utils import safe_as_int


def test_int_cast_not_possible():
    np.testing.assert_raises(ValueError, safe_as_int, 7.1)
    np.testing.assert_raises(ValueError, safe_as_int, [7.1, 0.9])
    np.testing.assert_raises(ValueError, safe_as_int, np.r_[7.1, 0.9])
    np.testing.assert_raises(ValueError, safe_as_int, (7.1, 0.9))
    np.testing.assert_raises(ValueError, safe_as_int, ((3,   4,   1),
                                                       (2, 7.6, 289)))

    np.testing.assert_raises(ValueError, safe_as_int, 7.1, 0.09)
    np.testing.assert_raises(ValueError, safe_as_int, [7.1, 0.9], 0.09)
    np.testing.assert_raises(ValueError, safe_as_int, np.r_[7.1, 0.9], 0.09)
    np.testing.assert_raises(ValueError, safe_as_int, (7.1, 0.9), 0.09)
    np.testing.assert_raises(ValueError, safe_as_int, ((3,   4,   1),
                                                       (2, 7.6, 289)), 0.25)


def test_int_cast_possible():
    np.testing.assert_equal(safe_as_int(7.1, atol=0.11), 7)
    np.testing.assert_equal(safe_as_int(-7.1, atol=0.11), -7)
    np.testing.assert_equal(safe_as_int(41.9, atol=0.11), 42)
    np.testing.assert_array_equal(safe_as_int([2, 42, 5789234.0, 87, 4]),
                                  np.r_[2, 42, 5789234, 87, 4])
    np.testing.assert_array_equal(safe_as_int(np.r_[[[3, 4,  1.000000001],
                                                     [7, 2, -8.999999999],
                                                     [6, 9, -4234918347.]]]),
                                  np.r_[[[3, 4,           1],
                                         [7, 2,          -9],
                                         [6, 9, -4234918347]]])


if __name__ == '__main__':
    np.testing.run_module_suite()
