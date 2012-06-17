import numpy as np
from numpy.testing import assert_equal, assert_array_equal
from nose.tools import assert_true, assert_greater
from skimage.segmentation import quickshift


def test_grey():
    rnd = np.random.RandomState(0)
    img = np.zeros((20, 20))
    img[:10, :10] = 0.2
    img[10:, :10] = 0.4
    img[10:, 10:] = 0.6
    img += 0.1 * rnd.normal(size=img.shape)
    seg = quickshift(img, random_seed=0)
    # we expect 4 segments:
    assert_equal(len(np.unique(seg)), 4)
    # that mostly respect the 4 regions:
    for i in xrange(4):
        hist = np.histogram(img[seg == i], bins=[0, 0.1, 0.3, 0.5, 1])[0]
        assert_greater(hist[i], 40)


def test_color():
    rnd = np.random.RandomState(0)
    img = np.zeros((20, 20, 3))
    img[:10, :10, 0] = 1
    img[10:, :10, 1] = 1
    img[10:, 10:, 2] = 1
    img += 0.2 * rnd.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    seg = quickshift(img, random_seed=0)
    # we expect 4 segments:
    assert_equal(len(np.unique(seg)), 4)
    assert_array_equal(seg[:10, :10], 0)
    assert_array_equal(seg[10:, :10], 3)
    assert_array_equal(seg[:10, 10:], 1)
    assert_array_equal(seg[10:, 10:], 2)

    seg2 = quickshift(img, sigma=1, tau=3, random_seed=0)
    # very oversegmented:
    assert_equal(len(np.unique(seg2)), 30)
    # still don't cross lines
    assert_true((seg2[9, :] != seg2[10, :]).all())
    assert_true((seg2[:, 9] != seg2[:, 10]).all())


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
