import numpy as np
from numpy import pi, sin, cos
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal)

from skimage.transform._orientation import (compute_rotation_matrix,
                                            _normalize, _axis_0_rotation_matrix)


def test_normalize():
    """test that normalized vectors have magnitude of 1"""
    X = np.arange(10)
    uX = _normalize(X)

    assert_almost_equal(np.linalg.norm(uX), 1)


def test_axis_0_rotation_matrix():
    """test that rotation matrix properly rotates vector to face axis 0"""
    # trivial case
    M = _axis_0_rotation_matrix([0, 1, 0])
    assert_almost_equal(M @ [0, 1, 0], [1, 0, 0])

    # non-trivial case
    M = _axis_0_rotation_matrix([0, .5, .5])
    assert_almost_equal(M @ [0, .5, .5], [np.hypot(.5, .5), 0, 0])


def test_compute_rotation_matrix():
    """test that rotation matrix properly rotates X to coincide with Y"""
    # trivial case
    X = [1, 0, 0]
    Y = [0, 0, 1]

    M = compute_rotation_matrix(X, Y)

    Z = M @ X

    assert_almost_equal(Z, Y)

    # non-trivial case
    X = np.arange(5)
    Y = np.arange(5, 10)
    uY = np.linalg.norm(X) / np.linalg.norm(Y) * Y

    M = compute_rotation_matrix(X, Y)

    Z = M @ X

    assert_almost_equal(Z, uY)

    # ensure preserves expected length
    X = np.arange(0, 50)
    Y = np.arange(175, 225)

    M = compute_rotation_matrix(X, Y)

    Z = np.arange(250, 300)
    rotZ = M @ Z

    assert_almost_equal(np.linalg.norm(rotZ), np.linalg.norm(Z))


def test_compute_rotation_matrix_homogeneous():
    """test that rotation matrix properly handles homogeneity"""
    X = np.arange(5)
    Y = np.arange(5, 10)

    M = compute_rotation_matrix(X, Y, use_homogeneous_coords=True)

    assert_equal(M[-1], [0, 0, 0, 0, 1])
    assert_equal(M[:, -1], [0, 0, 0, 0, 1])

    Z = np.arange(100, 105)
    rotZ = M @ Z

    # since rotation matrix is homogenous, the rotated and original
    # vectors should share the same last coordinates
    assert_equal(rotZ[-1], Z[-1])
