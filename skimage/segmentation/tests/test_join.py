import numpy as np
from numpy.testing import assert_array_equal, assert_raises
from skimage.segmentation import join_segmentations, relabel_from_one

def test_join_segmentations():
    s1 = np.array([[0, 0, 1, 1],
                   [0, 2, 1, 1],
                   [2, 2, 2, 1]])
    s2 = np.array([[0, 1, 1, 0],
                   [0, 1, 1, 0],
                   [0, 1, 1, 1]])

    # test correct join
    # NOTE: technically, equality to j_ref is not required, only that there
    # is a one-to-one mapping between j and j_ref. I don't know of an easy way
    # to check this (i.e. not as error-prone as the function being tested)
    j = join_segmentations(s1, s2)
    j_ref = np.array([[0, 1, 3, 2],
                      [0, 5, 3, 2],
                      [4, 5, 5, 3]])
    assert_array_equal(j, j_ref)

    # test correct exception when arrays are different shapes
    s3 = np.array([[0, 0, 1, 1], [0, 2, 2, 1]])
    assert_raises(ValueError, join_segmentations, s1, s3)

def test_relabel_from_one():
    ar = np.array([1, 1, 5, 5, 8, 99, 42])
    ar_relab, fw, inv = relabel_from_one(ar)
    ar_relab_ref = np.array([1, 1, 2, 2, 3, 5, 4])
    assert_array_equal(ar_relab, ar_relab_ref)
    fw_ref = np.zeros(100, int)
    fw_ref[1] = 1; fw_ref[5] = 2; fw_ref[8] = 3; fw_ref[42] = 4; fw_ref[99] = 5
    assert_array_equal(fw, fw_ref)
    inv_ref = np.array([0,  1,  5,  8, 42, 99])
    assert_array_equal(inv, inv_ref)


if __name__ == "__main__":
    np.testing.run_module_suite()
