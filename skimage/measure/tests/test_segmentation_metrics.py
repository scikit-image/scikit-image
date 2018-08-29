import numpy as np

from skimage.measure import (compare_adapted_rand_error,
                             compare_raw_edit_distance,
                             compare_split_variation_of_information,
                             compare_variation_of_information)

from skimage._shared.testing import assert_equal, assert_almost_equal

def test_red():
    im_true = np.array([[2, 1], [1, 2]])
    im_test = np.array([[1, 2],[3, 1]])
    assert_equal(compare_raw_edit_distance(im_true, im_test), (-3.0, 0.0))

def test_vi():
    im_true = np.array([1, 2, 3, 4])
    im_test = np.array([1, 1, 8, 8])
    assert_equal(compare_variation_of_information(im_true, im_test), -1)

def test_are():
    im_true = np.array([[2, 1], [1, 2]])
    im_test = np.array([[1, 2],[3, 1]])
    assert_almost_equal(compare_adapted_rand_error(im_true, im_test), (0.3333333, 0.5, 1.0))
