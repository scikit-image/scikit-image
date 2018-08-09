import numpy as np
from skimage._shared.testing import assert_equal
from skimage._shared import testing

from skimage.feature import mnxc
from skimage.data import camera

def test_mnxc_output_shape():
    """Masked normalized cross-correlation should return a shape of N + M + 1 for each transform axis."""
    shape1 = (15, 4, 5)
    shape2 = (6, 12, 7)
    expected_full_shape = tuple(np.array(shape1) + np.array(shape2) - 1)
    expected_same_shape = shape1

    arr1 = np.zeros(shape1)
    arr2 = np.zeros(shape2)
    # Trivial masks
    m1 = np.ones_like(arr1)
    m2 = np.ones_like(arr2)

    full_xcorr = mnxc(arr1, arr2, m1, m2, axes = (0, 1, 2), mode = 'full')
    assert_equal(full_xcorr.shape, expected_full_shape)

    same_xcorr = mnxc(arr1, arr2, m1, m2, axes = (0, 1, 2), mode = 'same')
    assert_equal(same_xcorr.shape, expected_same_shape)

def test_mnxc_test_against_mismatched_dimensions():
    """Masked normalized cross-correlation should raise an error if array dimensions along non-transformation axes are mismatched."""
    shape1 = (23, 1, 1)
    shape2 = (6, 2, 2)

    arr1 = np.zeros(shape1)
    arr2 = np.zeros(shape2)

    # Trivial masks
    m1 = np.ones_like(arr1)
    m2 = np.ones_like(arr2)

    with testing.raises(ValueError):
        mnxc(arr1, arr2, m1, m2, axes = (1, 2))

def test_mnxc_output_range():
    """Masked normalized cross-correlation should return between 1 and -1."""
    # See random number generator for reproducible results
    np.random.seed(23)

    # Array dimensions must match along non-transformation axes, in this case axis 0
    shape1 = (15, 4, 5)
    shape2 = (15, 12, 7)

    # Initial array ranges between -5 and 5
    arr1 = 10*np.random.random(shape1) - 5
    arr2 = 10*np.random.random(shape2) - 5

    # random masks
    m1 = np.random.choice([True, False], arr1.shape)
    m2 = np.random.choice([True, False], arr2.shape)

    xcorr = mnxc(arr1, arr2, m1, m2, axes = (1, 2))

    # No assert array less or equal, so we add an eps
    # Also could not find an `assert_array_greater`, Use (-xcorr) instead
    eps = np.finfo(np.float).eps
    testing.assert_array_less(xcorr, 1 + eps)
    testing.assert_array_less(-xcorr, 1 + eps)

def test_mnxc_side_effects():
    """Masked normalized cross-correlation should not modify the inputs."""
    shape1 = (2, 2, 2)
    shape2 = (2, 2, 2)

    arr1 = np.zeros(shape1)
    arr2 = np.zeros(shape2)

    # Trivial masks
    m1 = np.ones_like(arr1)
    m2 = np.ones_like(arr2)

    for arr in (arr1, arr2, m1, m2):
        arr.setflags(write = False)
    
    mnxc(arr1, arr2, m1, m2)

def test_mnxc_over_axes():
    """Masked normalized cross-correlation over axes should be equivalent to a loop over non-transform axes."""
    # See random number generator for reproducible results
    np.random.seed(23)

    arr1 = np.random.random((8, 8, 5))
    arr2 = np.random.random((8, 8, 5))

    m1 = np.random.choice([True, False], arr1.shape)
    m2 = np.random.choice([True, False], arr2.shape)

    # Loop over last axis
    with_loop = np.empty_like(arr1, dtype = np.complex)
    for index in range(arr1.shape[-1]):
        with_loop[:,:,index] = mnxc(arr1[:,:,index], arr2[:,:,index], 
                                    m1[:,:,index], m2[:,:,index], axes = (0, 1), mode = 'same')
    
    over_axes = mnxc(arr1, arr2, m1, m2, axes = (0, 1), mode = 'same')

    testing.assert_array_almost_equal(with_loop, over_axes)

def test_mnxc_autocorrelation_trivial_masks():
    """Masked normalized cross-correlation between identical arrays should reduce to an autocorrelation even with random masks."""
    # See random number generator for reproducible results
    np.random.seed(23)

    arr1 = camera()
    arr2 = camera()
    
    # Random masks with 75% of pixels being valid
    m1 = np.random.choice([True, False], arr1.shape, p = [3/4, 1/4])
    m2 = np.random.choice([True, False], arr2.shape, p = [3/4, 1/4])

    xcorr = mnxc(arr1, arr1, m1, m1, axes = (0, 1), mode = 'same', overlap_ratio = 0).real
    max_index = np.unravel_index(np.argmax(xcorr), xcorr.shape)

    # Autocorrelation should have maximum in center of array
    testing.assert_almost_equal(xcorr.max(), 1)
    testing.assert_array_equal(max_index, np.array(arr1.shape) / 2)
