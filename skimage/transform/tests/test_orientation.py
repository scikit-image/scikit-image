import numpy as np
from numpy import pi, sin, cos
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal)

from skimage.transform._orientation import (convert_quasipolar_coords,
                                            compute_rotation_matrix,
                                            _normalize, _axis_0_rotation_matrix)


def test_normalize():
    X = np.arange(10)
    uX = _normalize(X)

    assert_almost_equal(np.linalg.norm(uX), 1)


def test_axis_0_rotation_matrix():
    pass


def test_convert_quasipolar_coords():
    # test polar case
    y, x = convert_quasipolar_coords(2, [pi / 4])
    assert_almost_equal([x, y], [np.sqrt(2), np.sqrt(2)])

    # test spherical case
    y, x, z = convert_quasipolar_coords(10, [pi / 4, pi / 2])
    assert_almost_equal([x, y, z], [10 * sin(pi / 2) * cos(pi / 4),
                                    10 * sin(pi / 2) * sin(pi / 4),
                                    10 * cos(pi / 2)])

    # test higher-dimensional case
    coords = convert_quasipolar_coords(1, [pi / 3,
                                           5 * pi / 6,
                                           3 * pi / 4,
                                           pi / 6,
                                           pi / 4])
    assert_almost_equal(coords, [sin(pi / 3)
                                 * sin(5 * pi / 6)
                                 * sin(3 * pi / 4)
                                 * sin(pi / 6)
                                 * sin(pi / 4),
                                 cos(pi / 3)
                                 * sin(5 * pi / 6)
                                 * sin(3 * pi / 4)
                                 * sin(pi / 6)
                                 * sin(pi / 4),
                                 cos(5 * pi / 6)
                                 * sin(3 * pi / 4)
                                 * sin(pi / 6)
                                 * sin(pi / 4),
                                 cos(3 * pi / 4)
                                 * sin(pi / 6)
                                 * sin(pi / 4),
                                 cos(pi / 6)
                                 * sin(pi / 4),
                                 cos(pi / 4)])


def test_compute_rotation_matrix():
    # trivial case
    X = [1, 0, 0]
    Y = [0, 0, 1]

    M = compute_rotation_matrix(X, Y)

    Z = M @ X

    assert_almost_equal(Z, Y)

    # complex case
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
    X = np.arange(5)
    Y = np.arange(5, 10)

    M = compute_rotation_matrix(X, Y, use_homogeneous_coords=True)

    assert_equal(M[-1], [0, 0, 0, 0, 1])
    assert_equal(M[:, -1], [0, 0, 0, 0, 1])

    Z = np.arange(100, 105)
    rotZ = M @ Z

    assert_equal(rotZ[-1], Z[-1])


def test_compute_quasipolar_rotation_matrix():
    # 2D case
    theta = pi / 6

    R = [[ cos(theta), -sin(theta) ],
         [ sin(theta),  cos(theta) ]]

    y, x = convert_quasipolar_coords(1, [theta])
    M = compute_rotation_matrix([1, 0], [x, y])

    assert_almost_equal(M, R)

    # 3D case
    theta = pi / 3
    phi = pi / 4

    R = [[ cos(theta) * cos(phi), -sin(theta), cos(theta) * -sin(phi) ],
         [ sin(theta) * cos(phi),  cos(theta), sin(theta) * -sin(phi) ],
         [              sin(phi),           0,               cos(phi) ]]

    y, x, z = convert_quasipolar_coords(1, [theta, phi])
    M = compute_rotation_matrix([1, 0, 0], [x, y, z])

    assert_almost_equal(M, R)
