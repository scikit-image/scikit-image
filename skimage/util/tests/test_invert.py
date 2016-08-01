import numpy as np
from numpy.testing import assert_array_equal
from skimage import dtype_limits
from skimage.util.invert import invert


def test_invert_bool():
    dtype = 'bool'
    image = np.zeros((3, 3), dtype=dtype)
    image[1, :] = dtype_limits(image)[1]
    expected = np.zeros((3, 3), dtype=dtype) + dtype_limits(image)[1]
    expected[1, :] = 0
    result = invert(image)
    assert_array_equal(expected, result)


def test_invert_uint8():
    dtype = 'uint8'
    image = np.zeros((3, 3), dtype=dtype)
    image[1, :] = dtype_limits(image)[1]
    expected = np.zeros((3, 3), dtype=dtype) + dtype_limits(image)[1]
    expected[1, :] = 0
    result = invert(image)
    assert_array_equal(expected, result)


def test_invert_int8():
    dtype = 'int8'
    image = np.zeros((3, 3), dtype=dtype)
    image[1, :] = dtype_limits(image)[1]
    expected = np.zeros((3, 3), dtype=dtype) + dtype_limits(image)[1]
    expected[1, :] = 0
    result = invert(image)
    assert_array_equal(expected, result)


def test_invert_float64():
    dtype = 'float64'
    image = np.zeros((3, 3), dtype=dtype)
    image[1, :] = dtype_limits(image)[1]
    expected = np.zeros((3, 3), dtype=dtype) + dtype_limits(image)[1]
    expected[1, :] = 0
    result = invert(image)
    assert_array_equal(expected, result)


if __name__ == '__main__':
    np.testing.run_module_suite()
