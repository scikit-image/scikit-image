import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal)

from skimage.filters._gabor import (gabor_kernel, gabor, _sigma_prefactor,
                                    _convert_quasipolar_coords, _normalize,
                                    _compute_rotation_matrix)


def _gabor_2d(frequency, theta=0, bandwidth=1, sigma_x=None, sigma_y=None,
              n_stds=3, offset=0):
    if sigma_x is None:
        sigma_x = _sigma_prefactor(bandwidth) / frequency
    if sigma_y is None:
        sigma_y = _sigma_prefactor(bandwidth) / frequency

    x0 = np.ceil(max(np.abs(n_stds * sigma_x * np.cos(theta)),
                     np.abs(n_stds * sigma_y * np.sin(theta)), 1))
    y0 = np.ceil(max(np.abs(n_stds * sigma_y * np.cos(theta)),
                     np.abs(n_stds * sigma_x * np.sin(theta)), 1))
    y, x = np.mgrid[-y0:y0 + 1, -x0:x0 + 1]

    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)

    g = np.zeros(y.shape, dtype=np.complex)
    g[:] = np.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2))
    g /= 2 * np.pi * sigma_x * sigma_y
    g *= np.exp(1j * (2 * np.pi * frequency
                      * (x * np.cos(theta) + y * np.sin(theta))
                      + offset))

    return g


def _gabor_3d(frequency, alpha=0, beta=0, bandwidth=1, sigma_x=None,
              sigma_y=None, sigma_z=None, n_stds=3, offset=0):
    if sigma_x is None:
        sigma_x = _sigma_prefactor(bandwidth) / frequency
    if sigma_y is None:
        sigma_y = _sigma_prefactor(bandwidth) / frequency
    if sigma_z is None:
        sigma_z = _sigma_prefactor(bandwidth) / frequency

    """
    R = [[ cos(alpha) * sin(beta), -sin(alpha), -cos(alpha) * cos(beta) ],
         [ sin(alpha) * sin(beta),  cos(alpha), -sin(alpha) * cos(beta) ],
         [              cos(beta),           0,               sin(beta) ]]
    """
    x0 = np.ceil(max(np.abs(n_stds * sigma_x * np.cos(alpha) * np.sin(beta)),
                     np.abs(n_stds * sigma_y * np.sin(alpha)),
                     np.abs(n_stds * sigma_z * np.cos(alpha) * np.cos(beta)),
                     1))
    y0 = np.ceil(max(np.abs(n_stds * sigma_x * np.sin(alpha) * np.sin(beta)),
                     np.abs(n_stds * sigma_y * np.cos(alpha)),
                     np.abs(n_stds * sigma_z * np.sin(alpha) * np.cos(beta)),
                     1))
    z0 = np.ceil(max(np.abs(n_stds * sigma_x * np.cos(beta)),
                     np.abs(n_stds * sigma_z * np.sin(beta)),
                     1))
    y, x, z = np.mgrid[-y0:y0 + 1, -x0:x0 + 1, -z0:z0 + 1]

    rotx = (x * np.cos(alpha) * np.sin(beta)
            + y * np.sin(alpha) * np.sin(beta)
            + z * np.cos(beta))
    roty = -x * np.sin(alpha) + y * np.cos(alpha)
    rotz = (-x * np.cos(alpha) * np.cos(beta)
            - y * np.sin(alpha) * np.cos(beta)
            + z * np.sin(beta))

    g = np.zeros(y.shape, dtype=np.complex)
    g[:] = np.exp(-0.5 * (rotx ** 2 / sigma_x ** 2
                          + roty ** 2 / sigma_y ** 2
                          + rotz ** 2 / sigma_z ** 2))
    g /= (2 * np.pi) ** (3 / 2) * sigma_x * sigma_y * sigma_z
    g *= np.exp(1j * (2 * np.pi * frequency
                      * (x * np.sin(beta) * np.cos(alpha)
                         + y * np.sin(beta) * np.sin(alpha)
                         + z * np.cos(beta))
                      + offset))

    return g


def test_gabor_kernel_2d():
    for theta in np.arange(0, np.pi, .2):
        for sigma_x in range(1, 10, 3):
            for sigma_y in range(1, 10, 3):
                for frequency in np.arange(0, 2, .3):
                    kernel_nd = gabor_kernel(frequency, theta=theta,
                                             sigma=(sigma_y, sigma_x))
                    kernel_2d = _gabor_2d(frequency, theta=theta,
                                          sigma_x=sigma_x, sigma_y=sigma_y)
                    assert_almost_equal(kernel_nd, kernel_2d)


def test_gabor_kernel_3d():
    for alpha in np.arange(0, np.pi, .2):
        for beta in np.arange(0, np.pi, .2):
            for sigma_x in range(1, 10, 3):
                for sigma_y in range(1, 10, 3):
                    for sigma_z in range(1, 10, 3):
                        for frequency in np.arange(0, 2, .3):
                            kernel_nd = gabor_kernel(frequency,
                                                     theta=(alpha, beta),
                                                     sigma=(sigma_y,
                                                            sigma_x,
                                                            sigma_z),
                                                     ndim=3)
                            kernel_3d = _gabor_3d(frequency, alpha=alpha,
                                                  beta=beta, sigma_x=sigma_x,
                                                  sigma_y=sigma_y,
                                                  sigma_z=sigma_z)
                            assert_almost_equal(kernel_nd, kernel_3d)


def test_gabor_kernel_size():
    n_stds = 3
    sigma_x = 5
    sigma_y = 10
    # Sizes cut off at +/- three sigma + 1 for the center
    size_x = n_stds * sigma_x * 2 + 1
    size_y = n_stds * sigma_y * 2 + 1

    kernel = gabor_kernel(0, theta=0, sigma=(sigma_y, sigma_x), n_stds=n_stds)
    assert_equal(kernel.shape, (size_y, size_x))

    kernel = gabor_kernel(0, theta=np.pi / 2, sigma=(sigma_y, sigma_x),
                          n_stds=n_stds)
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


def test_convert_quasipolar_coords():
    s = np.sin
    c = np.cos

    def p(n, d):
        return n * np.pi / d

    # test polar case
    y, x = _convert_quasipolar_coords(2, (p(1, 4),))
    assert_almost_equal([x, y], [np.sqrt(2), np.sqrt(2)])

    # test spherical case
    y, x, z = _convert_quasipolar_coords(10, (p(1, 4), p(1, 2)))
    assert_almost_equal([x, y, z], [10 * s(p(1, 2)) * c(p(1, 4)),
                                    10 * s(p(1, 2)) * s(p(1, 4)),
                                    10 * c(p(1, 2))])

    # test higher-dimensional case
    coords = _convert_quasipolar_coords(1, (p(1, 3),
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

    Z = M @ X

    assert_almost_equal(Z, Y)

    # complex case
    X = np.arange(5)
    Y = np.arange(5, 10)
    uY = np.linalg.norm(X) / np.linalg.norm(Y) * Y

    M = _compute_rotation_matrix(X, Y)

    Z = M @ X

    assert_almost_equal(Z, uY)

    # ensure preserves expected length
    X = np.arange(0, 50)
    Y = np.arange(175, 225)

    M = _compute_rotation_matrix(X, Y)

    Z = np.arange(250, 300)
    rotZ = M @ Z

    assert_almost_equal(np.linalg.norm(rotZ), np.linalg.norm(Z))


def test_compute_rotation_matrix_homogeneous():
    X = np.arange(5)
    Y = np.arange(5, 10)

    M = _compute_rotation_matrix(X, Y, use_homogeneous_coords=True)

    assert_equal(M[-1], [0, 0, 0, 0, 1])
    assert_equal(M[:, -1], [0, 0, 0, 0, 1])

    Z = np.arange(100, 105)
    rotZ = M @ Z

    assert_equal(rotZ[-1], Z[-1])


def test_compute_rotation_matrix_quasipolar():
    c = np.cos
    s = np.sin

    # 2D case
    X = np.asarray([1, 0])
    y, x = _convert_quasipolar_coords(5, [np.pi / 6])
    Y = np.asarray([x, y])

    M = _compute_rotation_matrix(X, Y)

    assert_almost_equal(M,
                        [[ c(np.pi / 6), -s(np.pi / 6) ],
                         [ s(np.pi / 6),  c(np.pi / 6) ]])

    # 3D case
    alpha = np.pi / 3
    beta = np.pi / 6

    R = [[ c(alpha) * s(beta), -s(alpha), -c(alpha) * c(beta) ],
         [ s(alpha) * s(beta),  c(alpha), -s(alpha) * c(beta) ],
         [            c(beta),         0,             s(beta) ]]


    X = np.asarray([1, 0, 0])
    y, x, z = _convert_quasipolar_coords(1, [alpha, beta])
    Y = np.asarray([x, y, z])

    M = _compute_rotation_matrix(X, Y)

    assert_almost_equal(M, R)


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
