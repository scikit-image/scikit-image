import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal)

from skimage.transform._orientation import (convert_quasipolar_coords,
                                            compute_rotation_matrix,
                                            compute_angular_rotation_matrix,
                                            _normalize, _axis_0_rotation_matrix)


def test_normalize():
    X = np.arange(10)
    uX = _normalize(X)

    assert_almost_equal(np.linalg.norm(uX), 1)


def test_axis_0_rotation_matrix():
    pass


def test_convert_quasipolar_coords():
    s = np.sin
    c = np.cos

    def p(n, d):
        return n * np.pi / d

    # test polar case
    y, x = convert_quasipolar_coords(2, (p(1, 4),))
    assert_almost_equal([x, y], [np.sqrt(2), np.sqrt(2)])

    # test spherical case
    y, x, z = convert_quasipolar_coords(10, (p(1, 4), p(1, 2)))
    assert_almost_equal([x, y, z], [10 * s(p(1, 2)) * c(p(1, 4)),
                                    10 * s(p(1, 2)) * s(p(1, 4)),
                                    10 * c(p(1, 2))])

    # test higher-dimensional case
    coords = convert_quasipolar_coords(1, (p(1, 3),
                                              p(5, 6),
                                              p(3, 4),
                                              p(1, 6),
                                              p(1, 4)))
    assert_almost_equal(coords, [s(p(1, 3))
                                 * s(p(5, 6))
                                 * s(p(3, 4))
                                 * s(p(1, 6))
                                 * s(p(1, 4)),
                                 c(p(1, 3))
                                 * s(p(5, 6))
                                 * s(p(3, 4))
                                 * s(p(1, 6))
                                 * s(p(1, 4)),
                                 c(p(5, 6))
                                 * s(p(3, 4))
                                 * s(p(1, 6))
                                 * s(p(1, 4)),
                                 c(p(3, 4))
                                 * s(p(1, 6))
                                 * s(p(1, 4)),
                                 c(p(1, 6))
                                 * s(p(1, 4)),
                                 c(p(1, 4))])


def test_compute_rotation_matrix():
    # trivial case
    X = np.asarray([1, 0, 0])
    Y = np.asarray([0, 0, 1])

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


def test_compute_angular_rotation_matrix():
    M = compute_angular_rotation_matrix((np.pi / 6,), axes=1)

    assert_almost_equal(M,
                        [[np.cos(np.pi / 6), -np.sin(np.pi / 6)],
                         [np.sin(np.pi / 6), np.cos(np.pi / 6)]])
