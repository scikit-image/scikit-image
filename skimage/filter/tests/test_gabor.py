import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from skimage.filter import gabor_kernel, gabor_filter


def test_gabor_kernel_sum():
    for sigmax in range(1, 10, 2):
        for sigmay in range(1, 10, 2):
            for frequency in range(0, 10, 2):
                kernel = gabor_kernel(sigmax, sigmay, frequency+0.1, 0)
                # make sure gaussian distribution is covered nearly 100%
                assert_almost_equal(np.abs(kernel).sum(), 1, 2)


def test_gabor_kernel_theta():
    for sigmax in range(1, 10, 2):
        for sigmay in range(1, 10, 2):
            for frequency in range(0, 10, 2):
                for theta in range(0, 10, 2):
                    kernel0 = gabor_kernel(sigmax, sigmay, frequency+0.1, theta)
                    kernel180 = gabor_kernel(sigmax, sigmay, frequency,
                                             theta+np.pi)

                    assert_array_almost_equal(np.abs(kernel0),
                                              np.abs(kernel180))


def test_gabor_filter():
    real, imag = gabor_filter(np.random.random((100, 100)), 1, 1, 1, 1)


if __name__ == "__main__":
    from numpy import testing
    testing.run_module_suite()
