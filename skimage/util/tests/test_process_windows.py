import numpy as np
from nose.tools import raises
from numpy.testing import assert_equal
from skimage.util.process_windows import process_windows


@raises(TypeError)
def test_process_windows_block_not_a_tuple():

    A = np.arange(10)
    process_windows(A, (5,), np.sum, [-1])    


@raises(ValueError)
def test_process_windows_wrong_block_dimension():

    A = np.arange(10)
    process_windows(A, (2, 2), np.sum)


@raises(ValueError)
def test_process_windows_1D_array_wrong_block_shape():

    A = np.arange(10)
    process_windows(A, (20,), np.sum)


def test_process_windows_1D_array():

    A = np.arange(10)
    B = process_windows(A, (1,), np.sum)
    assert_equal(B, np.arange(10))

def test_process_windows_2D_array_kwargs():

    A = np.arange(3 * 3).reshape([3,3])
    B = process_windows(A, (2,2), np.sum, {'axis':1})
    assert_equal(B, np.array([[[1,7], [3,9]],
                              [[7,13], [9,15]]]))


if __name__ == '__main__':
    np.testing.run_module_suite()
