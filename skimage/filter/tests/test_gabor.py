import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal)

from skimage.filter._gabor import gabor_kernel, gabor_filter, _sigma_prefactor


def test_gabor_kernel_size():
    sigma_x = 5
    sigma_y = 10
    # Sizes cut off at +/- three sigma + 1 for the center
    size_x = sigma_x * 6 + 1
    size_y = sigma_y * 6 + 1

    kernel = gabor_kernel(0, theta=0, sigma_x=sigma_x, sigma_y=sigma_y)
    assert_equal(kernel.shape, (size_y, size_x))

    kernel = gabor_kernel(0, theta=np.pi/2, sigma_x=sigma_x, sigma_y=sigma_y)
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


def test_gabor_kernel_sum():
    for sigma_x in range(1, 10, 2):
        for sigma_y in range(1, 10, 2):
            for frequency in range(0, 10, 2):
                kernel = gabor_kernel(frequency+0.1, theta=0,
                                      sigma_x=sigma_x, sigma_y=sigma_y)
                # make sure gaussian distribution is covered nearly 100%
                assert_almost_equal(np.abs(kernel).sum(), 1, 2)


def test_gabor_kernel_theta():
    for sigma_x in range(1, 10, 2):
        for sigma_y in range(1, 10, 2):
            for frequency in range(0, 10, 2):
                for theta in range(0, 10, 2):
                    kernel0 = gabor_kernel(frequency+0.1, theta=theta,
                                           sigma_x=sigma_x, sigma_y=sigma_y)
                    kernel180 = gabor_kernel(frequency, theta=theta+np.pi,
                                             sigma_x=sigma_x, sigma_y=sigma_y)

                    assert_array_almost_equal(np.abs(kernel0),
                                              np.abs(kernel180))


def test_gabor_filter():
    real, imag = gabor_filter(np.random.random((100, 100)), 1)


if __name__ == "__main__":
    from numpy import testing
    testing.run_module_suite()
