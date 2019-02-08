
import numpy as np

from skimage._shared import testing
from skimage._shared.testing import assert_array_almost_equal
from skimage.filters import threshold_local, gaussian
from skimage.util.apply_parallel import apply_parallel, dask_available, \
                                        check_parallel
from skimage.morphology import binary_erosion
from skimage.measure import label


@testing.skipif(not dask_available, reason="dask not installed")
def test_apply_parallel():
    import dask.array as da

    # data
    a = np.arange(144).reshape(12, 12).astype(float)

    # apply the filter
    expected1 = threshold_local(a, 3)
    result1 = apply_parallel(threshold_local, a, chunks=(6, 6), depth=5,
                             extra_arguments=(3,),
                             extra_keywords={'mode': 'reflect'})

    assert_array_almost_equal(result1, expected1)

    def wrapped_gauss(arr):
        return gaussian(arr, 1, mode='reflect')

    expected2 = gaussian(a, 1, mode='reflect')
    result2 = apply_parallel(wrapped_gauss, a, chunks=(6, 6), depth=5)

    assert_array_almost_equal(result2, expected2)

    expected3 = gaussian(a, 1, mode='reflect')
    result3 = apply_parallel(
        wrapped_gauss, da.from_array(a, chunks=(6, 6)), depth=5, compute=True
    )

    assert isinstance(result3, np.ndarray)
    assert_array_almost_equal(result3, expected3)


@testing.skipif(not dask_available, reason="dask not installed")
def test_apply_parallel_lazy():
    import dask.array as da

    # data
    a = np.arange(144).reshape(12, 12).astype(float)
    d = da.from_array(a, chunks=(6, 6))

    # apply the filter
    expected1 = threshold_local(a, 3)
    result1 = apply_parallel(threshold_local, a, chunks=(6, 6), depth=5,
                             extra_arguments=(3,),
                             extra_keywords={'mode': 'reflect'},
                             compute=False)

    # apply the filter on a Dask Array
    result2 = apply_parallel(threshold_local, d, depth=5,
                             extra_arguments=(3,),
                             extra_keywords={'mode': 'reflect'})

    assert isinstance(result1, da.Array)

    assert_array_almost_equal(result1.compute(), expected1)

    assert isinstance(result2, da.Array)

    assert_array_almost_equal(result2.compute(), expected1)


@testing.skipif(not dask_available, reason="dask not installed")
def test_no_chunks():
    a = np.ones(1 * 4 * 8 * 9).reshape(1, 4, 8, 9)

    def add_42(arr):
        return arr + 42

    expected = add_42(a)
    result = apply_parallel(add_42, a)

    assert_array_almost_equal(result, expected)


@testing.skipif(not dask_available, reason="dask not installed")
def test_apply_parallel_wrap():
    def wrapped(arr):
        return gaussian(arr, 1, mode='wrap')
    a = np.arange(144).reshape(12, 12).astype(float)
    expected = gaussian(a, 1, mode='wrap')
    result = apply_parallel(wrapped, a, chunks=(6, 6), depth=5, mode='wrap')

    assert_array_almost_equal(result, expected)


@testing.skipif(not dask_available, reason="dask not installed")
def test_apply_parallel_nearest():
    def wrapped(arr):
        return gaussian(arr, 1, mode='nearest')
    a = np.arange(144).reshape(12, 12).astype(float)
    expected = gaussian(a, 1, mode='nearest')
    result = apply_parallel(wrapped, a, chunks=(6, 6), depth={0: 5, 1: 5},
                            mode='nearest')

    assert_array_almost_equal(result, expected)


@testing.skipif(not dask_available, reason="dask not installed")
def test_check_parallel():
    depth = check_parallel(gaussian, shape=(100, 100), verbose=False)
    assert depth == 4
    depth = check_parallel(gaussian, shape=(100, 100), verbose=False,
                                     extra_keywords={'sigma':2})
    assert depth == 8
    depth = check_parallel(gaussian, shape=(100, 100), depth_max=20,
                                     verbose=False, extra_keywords={'sigma':4})
    assert depth == 16
    with testing.raises(ValueError):
        depth = check_parallel(gaussian, shape=(100, 100),
                                     verbose=False, extra_keywords={'sigma':4})
    with testing.raises(ValueError):
        check_parallel(label, shape=(100, 100), verbose=False)
