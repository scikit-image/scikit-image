import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal

from _skimage2._shared.utils import _supported_float_type
from skimage.filters._gabor import _sigma_prefactor, gabor, gabor_kernel


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


@pytest.mark.parametrize('bandwidth', [0, -0.5, -1])
def test_gabor_kernel_invalid_bandwidth(bandwidth):
    # `bandwidth=0` makes `_sigma_prefactor`'s denominator exactly zero, and a
    # negative `bandwidth` makes it return a negative sigma, which is not a
    # meaningful standard deviation. Both are rejected up front now.
    with pytest.raises(ValueError, match='bandwidth'):
        gabor_kernel(0.1, bandwidth=bandwidth)
    with pytest.raises(ValueError, match='bandwidth'):
        gabor(np.zeros((10, 10)), frequency=0.1, bandwidth=bandwidth)


@pytest.mark.parametrize('sigma_kwargs', [{'sigma_x': 5.0}, {'sigma_y': 5.0}])
def test_gabor_kernel_invalid_bandwidth_with_one_sigma(sigma_kwargs):
    # Only one sigma is user-supplied, so the other is still derived from
    # `bandwidth` and the guard must fire. Previously `bandwidth=-1` here left
    # exactly one sigma negative, which flipped the sign of the whole kernel
    # rather than cancelling out.
    with pytest.raises(ValueError, match='bandwidth'):
        gabor_kernel(0.1, bandwidth=-1, **sigma_kwargs)
    with pytest.raises(ValueError, match='bandwidth'):
        gabor_kernel(0.1, bandwidth=0, **sigma_kwargs)


@pytest.mark.parametrize('bandwidth', [0, -1])
def test_gabor_kernel_bandwidth_ignored_when_sigmas_given(bandwidth):
    # The docstring says `bandwidth` is ignored when both sigmas are set, so the
    # new guard must not fire there.
    expected = gabor_kernel(0.1, sigma_x=1.0, sigma_y=1.0)
    kernel = gabor_kernel(0.1, bandwidth=bandwidth, sigma_x=1.0, sigma_y=1.0)
    assert_array_almost_equal(kernel, expected)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_gabor_kernel_dtype(dtype):
    kernel = gabor_kernel(1, bandwidth=1, dtype=dtype)
    assert kernel.dtype == dtype


@pytest.mark.parametrize('dtype', [np.uint8, np.float32])
def test_gabor_kernel_invalid_dtype(dtype):
    with pytest.raises(ValueError):
        kernel = gabor_kernel(1, bandwidth=1, dtype=dtype)
        assert kernel.dtype == dtype


def test_sigma_prefactor():
    assert_almost_equal(_sigma_prefactor(1), 0.56, 2)
    assert_almost_equal(_sigma_prefactor(0.5), 1.09, 2)


def test_gabor_kernel_sum():
    for sigma_x in range(1, 10, 2):
        for sigma_y in range(1, 10, 2):
            for frequency in range(0, 10, 2):
                kernel = gabor_kernel(
                    frequency + 0.1, theta=0, sigma_x=sigma_x, sigma_y=sigma_y
                )
                # make sure gaussian distribution is covered nearly 100%
                assert_almost_equal(np.abs(kernel).sum(), 1, 2)


def test_gabor_kernel_theta():
    for sigma_x in range(1, 10, 2):
        for sigma_y in range(1, 10, 2):
            for frequency in range(0, 10, 2):
                for theta in range(0, 10, 2):
                    kernel0 = gabor_kernel(
                        frequency + 0.1, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y
                    )
                    kernel180 = gabor_kernel(
                        frequency, theta=theta + np.pi, sigma_x=sigma_x, sigma_y=sigma_y
                    )

                    assert_array_almost_equal(np.abs(kernel0), np.abs(kernel180))


def test_gabor():
    Y, X = np.mgrid[:40, :40]
    frequencies = (0.1, 0.3)
    wave_images = [np.sin(2 * np.pi * X * f) for f in frequencies]

    def match_score(image, frequency):
        gabor_responses = gabor(image, frequency)
        return np.mean(np.hypot(*gabor_responses))

    # Gabor scores: diagonals are frequency-matched, off-diagonals are not.
    responses = np.array(
        [[match_score(image, f) for f in frequencies] for image in wave_images]
    )
    assert responses[0, 0] > responses[0, 1]
    assert responses[1, 1] > responses[0, 1]
    assert responses[0, 0] > responses[1, 0]
    assert responses[1, 1] > responses[1, 0]


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_gabor_float_dtype(dtype):
    image = np.ones((16, 16), dtype=dtype)
    y = gabor(image, 0.3)
    assert all(arr.dtype == _supported_float_type(image.dtype) for arr in y)


@pytest.mark.parametrize('dtype', [np.uint8, np.int32, np.intp])
def test_gabor_int_dtype(dtype):
    image = np.full((16, 16), 128, dtype=dtype)
    y = gabor(image, 0.3)
    assert all(arr.dtype == dtype for arr in y)
