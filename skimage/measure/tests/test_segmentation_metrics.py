import numpy as np

from skimage.measure import (compare_adapted_rand_error,
                             compare_raw_edit_distance,
                             compare_split_variation_of_information,
                             compare_variation_of_information)

from skimage._shared.testing import assert_equal, assert_almost_equal

def test_red():
    seg = np.array([[2, 1], [1, 2]])
    gt = np.array([[1, 2],[3, 1]])
    assert_equal(compare_raw_edit_distance(seg, gt), (-3.0, 0.0))

def test_vi():
    seg = np.array([1, 2, 3, 4])
    gt = np.array([1, 1, 8, 8])
    assert_equal(compare_variation_of_information(seg, gt), 1)

def test_are():
    seg = np.array([[2, 1], [1, 2]])
    gt = np.array([[1, 2],[3, 1]])
    assert_almost_equal((compare_adapted_rand_error(seg, gt)), 0.3333333)