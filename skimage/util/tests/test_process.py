import numpy as np
from numpy.testing import assert_array_almost_equal

from skimage.filters import threshold_adaptive, gaussian_filter
from skimage.util import process_chunks


def test_process_chunks():
    # data
    a = np.arange(144).reshape(12, 12).astype(float)

    # wrapp the function we're applying
    def wrapped_thresh(arr):
        return threshold_adaptive(arr, 3, mode='reflect')

    # apply the filter
    expected1 = threshold_adaptive(a, 3)
    result1 = process_chunks(wrapped_thresh, a, chunks=(6, 6),
                             depth=5)

    assert_array_almost_equal(result1, expected1)

    def wrapped_gauss(arr):
        return gaussian_filter(arr, 1, mode='reflect')

    expected2 = gaussian_filter(a, 1, mode='reflect')
    result2 = process_chunks(wrapped_gauss, a, chunks=(6, 6),
                             depth=5)

    assert_array_almost_equal(result2, expected2)


def test_no_chunks():
    a = np.ones(1 * 4 * 8 * 9).reshape(1, 4, 8, 9)

    def add_42(arr):
        return arr + 42

    expected = add_42(a)
    result = process_chunks(add_42, a)

    assert_array_almost_equal(result, expected)


def test_process_chunks_wrap():
    def wrapped(arr):
        return gaussian_filter(arr, 1, mode='wrap')
    a = np.arange(144).reshape(12, 12).astype(float)
    expected = gaussian_filter(a, 1, mode='wrap')
    result = process_chunks(wrapped, a, chunks=(6, 6),
                            depth=5, mode='wrap')

    assert_array_almost_equal(result, expected)


def test_process_chunks_nearest():
    def wrapped(arr):
        return gaussian_filter(arr, 1, mode='nearest')
    a = np.arange(144).reshape(12, 12).astype(float)
    expected = gaussian_filter(a, 1, mode='nearest')
    result = process_chunks(wrapped, a, chunks=(6, 6),
                            depth={0: 5, 1: 5}, mode='nearest')

    assert_array_almost_equal(result, expected)
