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

def test_process_windows_2D_array_args_with_overlap():
    A = np.arange(5 * 6).reshape((5, 6))
    B = process_blocks(A, (3, 4), np.sum, overlap=2)
    """ Currently:
        >>> A
        array([[ 0,  1,  2,  3,  4,  5],
               [ 6,  7,  8,  9, 10, 11],
               [12, 13, 14, 15, 16, 17],
               [18, 19, 20, 21, 22, 23],
               [24, 25, 26, 27, 28, 29]])
        >>> B.shape
        (2,2)
        >>> B
        array([[ 90, 114],
               [234, 258]])
        Expected:
        >>> B.shape
        (3,2)
        
    """
    
    

def test_process_windows_2D_array_args_with_2D_overlap():
    A = np.arange(5 * 6).reshape((5, 6))
    print "Allow n-dimensional overlap specifications?"
    B = process_blocks(A, (3, 4), np.sum, overlap=(2,2))
    


if __name__ == '__main__':
    np.testing.run_module_suite()
