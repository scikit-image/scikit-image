import numpy as np
from nose.tools import raises
from numpy.testing import assert_equal
from skimage.util import process_blocks


@raises(ValueError)
def test_process_windows_wrong_block_dimension():
    A = np.arange(10)
    process_blocks(A, (2, 2), np.sum)


def test_process_windows_1D_array():
    A = np.arange(10)
    B = process_blocks(A, (1,), np.sum)
    assert_equal(B, A)


def test_process_windows_2D_array_args():
    A = np.arange(4 * 4).reshape((4, 4))
    B = process_blocks(A, (2, 2), np.sum, {'axis': 1})
    assert_equal(B, np.array([[[1, 9], [5, 13]],
                              [[17, 25], [21, 29]]]))


if __name__ == '__main__':
    np.testing.run_module_suite()
