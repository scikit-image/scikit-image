import numpy as np
import scipy.sparse as sparse

from skimage.metrics import (adapted_rand_error,
                             variation_of_information,
                             contingency_table)

from skimage._shared.testing import assert_equal, assert_almost_equal, assert_array_equal


def test_contingency_table():
    im_true = np.array([1, 2, 3, 4])
    im_test = np.array([1, 1, 8, 8])

    table1 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0.25, 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0.25, 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0.25],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0.25]])

    sparse_table2 = contingency_table(im_true, im_test, normalize=True)
    table2 = sparse_table2.toarray()
    assert_array_equal(table1, table2)


def test_vi():
    im_true = np.array([1, 2, 3, 4])
    im_test = np.array([1, 1, 8, 8])
    assert_equal(np.sum(variation_of_information(im_true, im_test)), 1)


def test_are():
    im_true = np.array([[2, 1], [1, 2]])
    im_test = np.array([[1, 2], [3, 1]])
    assert_almost_equal(adapted_rand_error(im_true, im_test),
                        (0.3333333, 0.5, 1.0))
