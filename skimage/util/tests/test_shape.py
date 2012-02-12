import numpy as np
from nose.tools import raises
from numpy.testing import assert_equal
from skimage.util.shape import view_as_blocks, view_as_windows


@raises(TypeError)
def test_view_as_blocks_block_not_a_tuple():

    A = np.arange(10)
    view_as_blocks(A, [5])


@raises(ValueError)
def test_view_as_blocks_negative_shape():

    A = np.arange(10)
    view_as_blocks(A, (-2,))


@raises(ValueError)
def test_view_as_blocks_block_too_large():

    A = np.arange(10)
    view_as_blocks(A, (11,))


@raises(ValueError)
def test_view_as_blocks_wrong_block_dimension():

    A = np.arange(10)
    view_as_blocks(A, (2, 2))


@raises(ValueError)
def test_view_as_blocks_1D_array_wrong_block_shape():

    A = np.arange(10)
    view_as_blocks(A, (3,))


def test_view_as_blocks_1D_array():

    A = np.arange(10)
    B = view_as_blocks(A, (5,))
    assert_equal(B, np.array([[0, 1, 2, 3, 4],
                              [5, 6, 7, 8, 9]]))


def test_view_as_blocks_2D_array():

    A = np.arange(4 * 4).reshape(4, 4)
    B = view_as_blocks(A, (2, 2))
    assert_equal(B[0, 1], np.array([[2, 3],
                                   [6, 7]]))
    assert_equal(B[1, 0, 1, 1], 13)


def test_view_as_blocks_3D_array():

    A = np.arange(4 * 4 * 6).reshape(4, 4, 6)
    B = view_as_blocks(A, (1, 2, 2))
    assert_equal(B.shape, (4, 2, 3, 1, 2, 2))
    assert_equal(B[2:, 0, 2], np.array([[[[52, 53],
                                          [58, 59]]],
                                        [[[76, 77],
                                          [82, 83]]]]))


@raises(TypeError)
def test_view_as_windows_input_not_array():

    A = [1, 2, 3, 4, 5]
    view_as_windows(A, (2,))


@raises(TypeError)
def test_view_as_windows_window_not_tuple():

    A = np.arange(10)
    view_as_windows(A, [2])


@raises(ValueError)
def test_view_as_windows_wrong_window_dimension():

    A = np.arange(10)
    view_as_windows(A, (2, 2))


@raises(ValueError)
def test_view_as_windows_negative_window_length():

    A = np.arange(10)
    view_as_windows(A, (-1,))


@raises(ValueError)
def test_view_as_windows_window_too_large():

    A = np.arange(10)
    view_as_windows(A, (11,))


def test_view_as_windows_1D():

    A = np.arange(10)
    window_shape = (3,)
    B = view_as_windows(A, window_shape)
    assert_equal(B, np.array([[0, 1, 2],
                              [1, 2, 3],
                              [2, 3, 4],
                              [3, 4, 5],
                              [4, 5, 6],
                              [5, 6, 7],
                              [6, 7, 8],
                              [7, 8, 9]]))


def test_view_as_windows_2D():

    A = np.arange(5 * 4).reshape(5, 4)
    window_shape = (4, 3)
    B = view_as_windows(A, window_shape)
    assert_equal(B.shape, (2, 2, 4, 3))
    assert_equal(B, np.array([[[[0,  1,  2],
                                [4,  5,  6],
                                [8,  9, 10],
                                [12, 13, 14]],
                               [[1,  2,  3],
                                [5,  6,  7],
                                [9, 10, 11],
                                [13, 14, 15]]],
                              [[[4,  5,  6],
                                [8,  9, 10],
                                [12, 13, 14],
                                [16, 17, 18]],
                               [[5,  6,  7],
                                [9, 10, 11],
                                [13, 14, 15],
                                [17, 18, 19]]]]))
