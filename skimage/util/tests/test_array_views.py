import numpy as np
from nose.tools import raises
from numpy.testing import assert_equal
from skimage.util.array_views import block_view


@raises(ValueError)
def test_block_view_block_not_a_tuple():

    A = np.arange(10)
    block_view(A, [5])


@raises(ValueError)
def test_block_view_negative_shape():

    A = np.arange(10)
    block_view(A, (-2))


@raises(ValueError)
def test_block_view_block_too_large():

    A = np.arange(10)
    block_view(A, (11,))


@raises(ValueError)
def test_block_view_wrong_block_dimension():

    A = np.arange(10)
    block_view(A, (2,2))


@raises(ValueError)
def test_block_view_1D_array_wrong_block_shape():

    A = np.arange(10)
    block_view(A, (3,))


def test_block_view_1D_array():

    A = np.arange(10)
    B = block_view(A, (5,))
    assert_equal(B, np.array([[0, 1, 2, 3, 4],
                              [5, 6, 7, 8, 9]]))


def test_block_view_2D_array():

    A = np.arange(4*4).reshape(4,4)
    B = block_view(A, (2,2))
    assert_equal(B[0,1], np.array([[2, 3],
                                   [6, 7]]))
    assert_equal(B[1, 0, 1, 1], 13)


def test_block_view_3D_array():

    A = np.arange(4*4*6).reshape(4,4,6)
    B = block_view(A, (1,2,2))
    assert_equal(B.shape, (4, 2, 3, 1, 2, 2))
    assert_equal(B[2:, 0, 2], np.array([[[[52, 53],
                                          [58, 59]]],
                                        [[[76, 77],
                                          [82, 83]]]]))
