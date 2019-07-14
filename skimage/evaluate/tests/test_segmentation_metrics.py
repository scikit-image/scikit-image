import numpy as np

from skimage.evaluate import (adapted_rand_error,
                             variation_of_information)

from skimage._shared.testing import assert_equal, assert_almost_equal

def test_vi():
    im_true = np.array([1, 2, 3, 4])
    im_test = np.array([1, 1, 8, 8])
    assert_equal(np.sum(variation_of_information(im_true, im_test)), 1)

def test_are():
    im_true = np.array([[2, 1], [1, 2]])
    im_test = np.array([[1, 2],[3, 1]])
    assert_almost_equal(adapted_rand_error(im_true, im_test),
                        (0.3333333, 0.5, 1.0))