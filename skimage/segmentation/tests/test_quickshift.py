import numpy as np
from skimage.segmentation import quickshift

from skimage._shared.testing import (assert_greater, test_parallel,
                                     assert_equal, assert_array_equal)


@test_parallel()
def test_grey():
    rnd = np.random.RandomState(0)
    img = np.zeros((20, 21))
    img[:10, 10:] = 0.2
    img[10:, :10] = 0.4
    img[10:, 10:] = 0.6
    img += 0.1 * rnd.normal(size=img.shape)
    seg = quickshift(img, kernel_size=2, max_dist=3, random_seed=0,
                     convert2lab=False, sigma=0)
    # we expect 4 segments:
    assert_equal(len(np.unique(seg)), 4)
    # that mostly respect the 4 regions:
    for i in range(4):
        hist = np.histogram(img[seg == i], bins=[0, 0.1, 0.3, 0.5, 1])[0]
        assert_greater(hist[i], 20)

    seg4 = quickshift(img, random_seed=0, max_dist=1000, kernel_size=10,
                      sigma=0, ratio=0.5, full_search=True, return_tree=True,
                      convert2lab=False)
    dist_to_parent = np.array(seg4[2]).astype('uint32')
    # we expect 1 root, all the other pixels have a distance of 1:
    assert_equal(np.sum(dist_to_parent == 0), 1)
    assert_equal(np.sum(dist_to_parent == 1), 419)


def test_color():
    rnd = np.random.RandomState(0)
    img = np.zeros((20, 21, 3))
    img[:10, :10, 0] = 1
    img[10:, :10, 1] = 1
    img[10:, 10:, 2] = 1
    img += 0.01 * rnd.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    seg = quickshift(img, random_seed=0, max_dist=30, kernel_size=10, sigma=0)
    # we expect 4 segments:
    assert_equal(len(np.unique(seg)), 4)
    assert_array_equal(seg[:10, :10], 1)
    assert_array_equal(seg[10:, :10], 2)
    assert_array_equal(seg[:10, 10:], 0)
    assert_array_equal(seg[10:, 10:], 3)

    seg2 = quickshift(img, kernel_size=1, max_dist=2, random_seed=0,
                      convert2lab=False, sigma=0)
    # very oversegmented:
    assert_equal(len(np.unique(seg2)), 7)
    # still don't cross lines
    assert (seg2[9, :] != seg2[10, :]).all()
    assert (seg2[:, 9] != seg2[:, 10]).all()

    # tests for full_search option
    seg3 = quickshift(img, random_seed=0, max_dist=1000, kernel_size=10,
                      sigma=0, full_search=True)
    # we expect 1 segment:
    assert_equal(len(np.unique(seg3)), 1)
    assert_array_equal(seg3, 0)

    # tests for full_search option with the tree
    seg4 = quickshift(img, random_seed=0, max_dist=1000, kernel_size=10,
                      sigma=0, ratio=0.5, full_search=True, return_tree=True)
    dist_to_parent = np.array(seg4[2]).astype('uint32')
    # we expect 1 root, 3 root children and all the other pixels have a
    # distance of 1:
    assert_equal(np.sum(dist_to_parent == 0), 1)
    assert_equal(np.sum(dist_to_parent > 1), 3)
    assert_equal(np.sum(dist_to_parent == 1), 416)
