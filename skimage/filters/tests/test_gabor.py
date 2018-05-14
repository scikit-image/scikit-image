import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal)

from skimage.filters._gabor import (gabor_kernel, gabor, _sigma_prefactor,
                                    _decompose_quasipolar_coords, _normalize,
                                    _compute_rotation_matrix)


def test_gabor_kernel_size():
    sigma_x = 5
    sigma_y = 10
    # Sizes cut off at +/- three sigma + 1 for the center
    size_x = sigma_x * 6 + 1
    size_y = sigma_y * 6 + 1

    kernel = gabor_kernel(0, theta=0, sigma_x=sigma_x, sigma_y=sigma_y)
    assert_equal(kernel.shape, (size_y, size_x))

    kernel = gabor_kernel(0, theta=np.pi / 2, sigma_x=sigma_x, sigma_y=sigma_y)
    assert_equal(kernel.shape, (size_x, size_y))


def test_gabor_kernel_bandwidth():
    kernel = gabor_kernel(1, bandwidth=1)
    assert_equal(kernel.shape, (5, 5))

    kernel = gabor_kernel(1, bandwidth=0.5)
    assert_equal(kernel.shape, (9, 9))

    kernel = gabor_kernel(0.5, bandwidth=1)
    assert_equal(kernel.shape, (9, 9))


def test_sigma_prefactor():
    assert_almost_equal(_sigma_prefactor(1), 0.56, 2)
    assert_almost_equal(_sigma_prefactor(0.5), 1.09, 2)


def test_decompose_quasipolar_coords():
    s = np.sin
    c = np.cos

    def p(n, d):
        return n * np.pi / d

    # test polar case
    y, x = _decompose_quasipolar_coords(2, (p(1, 4),))
    assert_almost_equal([x, y], [np.sqrt(2), np.sqrt(2)])

    # test spehrical case
    y, x, z = _decompose_quasipolar_coords(10, (p(1, 4), p(1, 2)))
    assert_almost_equal([x, y, z], [10 * s(p(1, 2)) * c(p(1, 4)),
                                    10 * s(p(1, 2)) * s(p(1, 4)),
                                    10 * c(p(1, 2))])

    # test higher-dimensional case
    coords = _decompose_quasipolar_coords(1, (p(1, 3),
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


def test_normalize():
    X = np.arange(10)
    uX = _normalize(X)

    assert_almost_equal(np.linalg.norm(uX), 1)


def test_compute_rotation_matrix():
    # trivial case
    X = np.asarray([1, 0, 0])
    Y = np.asarray([0, 0, 1])

    M = _compute_rotation_matrix(X, Y)

    Z = np.matmul(M, X)

    assert_almost_equal(Z, Y)

    # complex case
    X = np.arange(5)
    Y = np.arange(5, 10)
    uY = np.linalg.norm(X) / np.linalg.norm(Y) * Y

    M = _compute_rotation_matrix(X, Y)

    Z = np.matmul(M, X)

    assert_almost_equal(Z, uY)

    # ensure preserves expected length
    X = np.arange(0, 50)
    Y = np.arange(175, 225)

    M = _compute_rotation_matrix(X, Y)

    Z = np.arange(250, 300)
    rotZ = np.matmul(M, Z)

    assert_almost_equal(np.linalg.norm(rotZ), np.linalg.norm(Z))


def test_compute_rotation_matrix_quasipolar():
    # 2D case
    X = np.asarray([1, 0])
    y, x = _decompose_quasipolar_coords(5, [np.pi / 6])
    Y = np.asarray([x, y])

    M = _compute_rotation_matrix(X, Y)

    assert_almost_equal(M,
                        [[np.cos(np.pi / 6), -np.sin(np.pi / 6)],
                         [np.sin(np.pi / 6), np.cos(np.pi / 6)]])

    # FIXME: actually check components of the matrix; this test is redundant
    # 3D case
    X = np.asarray([0, 1, 0])
    y, x, z = _decompose_quasipolar_coords(1, [np.pi / 3, np.pi / 6])
    Y = np.asarray([x, y, z])

    M = _compute_rotation_matrix(X, Y)

    Z = np.matmul(M, X)

    assert_almost_equal(Z, Y)


def test_gabor_kernel_sum():
    for sigma_x in range(1, 10, 2):
        for sigma_y in range(1, 10, 2):
            for frequency in range(0, 10, 2):
                kernel = gabor_kernel(frequency + 0.1, theta=0,
                                      sigma_x=sigma_x, sigma_y=sigma_y)
                # make sure gaussian distribution is covered nearly 100%
                assert_almost_equal(np.abs(kernel).sum(), 1, 2)


def test_gabor_kernel_theta():
    for sigma_x in range(1, 10, 2):
        for sigma_y in range(1, 10, 2):
            for frequency in range(0, 10, 2):
                for theta in range(0, 10, 2):
                    kernel0 = gabor_kernel(frequency + 0.1, theta=theta,
                                           sigma_x=sigma_x, sigma_y=sigma_y)
                    kernel180 = gabor_kernel(frequency, theta=theta + np.pi,
                                             sigma_x=sigma_x, sigma_y=sigma_y)

                    assert_array_almost_equal(np.abs(kernel0),
                                              np.abs(kernel180))


def test_gabor():
    Y, X = np.mgrid[:40, :40]
    frequencies = (0.1, 0.3)
    wave_images = [np.sin(2 * np.pi * X * f) for f in frequencies]

    def match_score(image, frequency):
        gabor_responses = gabor(image, frequency)
        return np.mean(np.hypot(*gabor_responses))

    # Gabor scores: diagonals are frequency-matched, off-diagonals are not.
    responses = np.array([[match_score(image, f) for f in frequencies]
                          for image in wave_images])
    assert responses[0, 0] > responses[0, 1]
    assert responses[1, 1] > responses[0, 1]
    assert responses[0, 0] > responses[1, 0]
    assert responses[1, 1] > responses[1, 0]


if __name__ == "__main__":
    from numpy import testing
    testing.run_module_suite()
