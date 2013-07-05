import numpy as np
from numpy.testing import assert_array_equal
from skimage.measure import block_sum


def test_block_sum():
    image1 = np.arange(4 * 6).reshape(4, 6)
    out1 = block_sum(image1, (2, 3))
    expected1 = np.array([[ 24,  42],
                          [ 96, 114]])
    assert_array_equal(expected1, out1)

    image2 = np.arange(5 * 8).reshape(5, 8)
    out2 = block_sum(image2, (3, 3))
    expected2 = np.array([[ 81, 108,  87],
                          [174, 192, 138]])
    assert_array_equal(expected2, out2)


if __name__ == "__main__":
    np.testing.run_module_suite()
