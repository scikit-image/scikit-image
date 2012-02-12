import numpy as np
from nose.tools import raises
from numpy.testing import assert_equal
from skimage.util.array_views import block_view, rolling_view


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


@raises(ValueError)
def test_rolling_view_input_not_array():

    A = [1, 2, 3, 4, 5]
    rolling_view(A, (2,))


@raises(ValueError)
def test_rolling_view_window_not_tuple():

    A = np.arange(10)
    rolling_view(A, [2])


@raises(ValueError)
def test_rolling_view_wrong_window_dimension():

    A = np.arange(10)
    rolling_view(A, (2,2))


@raises(ValueError)
def test_rolling_view_negative_window_length():

    A = np.arange(10)
    rolling_view(A, (-1,))


@raises(ValueError)
def test_rolling_view_window_too_large():

    A = np.arange(10)
    rolling_view(A, (11,))


def test_rolling_view_1D():

    A = np.arange(10)
    window_shape = (3,)
    B = rolling_view(A, window_shape)
    assert_equal(B, np.array([[0, 1, 2],
                              [1, 2, 3],
                              [2, 3, 4],
                              [3, 4, 5],
                              [4, 5, 6],
                              [5, 6, 7],
                              [6, 7, 8],
                              [7, 8, 9]]))


def test_rolling_view_2D():

    A = np.arange(5*4).reshape(5, 4)
    window_shape = (4, 3)
    B = rolling_view(A, window_shape)
    assert_equal(B.shape, (2, 2, 4, 3))
    assert_equal(B, np.array([[[[ 0,  1,  2],
                                [ 4,  5,  6],
                                [ 8,  9, 10],
                                [12, 13, 14]],
                               [[ 1,  2,  3],
                                [ 5,  6,  7],
                                [ 9, 10, 11],
                                [13, 14, 15]]],
                              [[[ 4,  5,  6],
                                [ 8,  9, 10],
                                [12, 13, 14],
                                [16, 17, 18]],
                               [[ 5,  6,  7],
                                [ 9, 10, 11],
                                [13, 14, 15],
                                [17, 18, 19]]]]))
