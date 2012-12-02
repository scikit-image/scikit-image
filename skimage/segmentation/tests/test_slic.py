import numpy as np
from numpy.testing import assert_equal, assert_array_equal
from skimage.segmentation import slic


def test_color():
    rnd = np.random.RandomState(0)
    img = np.zeros((20, 21, 3))
    img[:10, :10, 0] = 1
    img[10:, :10, 1] = 1
    img[10:, 10:, 2] = 1
    img += 0.01 * rnd.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    seg = slic(img, sigma=0, n_segments=4)
    # we expect 4 segments:
    print(seg)
    assert_equal(len(np.unique(seg)), 4)
    assert_array_equal(seg[:10, :10], 0)
    assert_array_equal(seg[10:, :10], 2)
    assert_array_equal(seg[:10, 10:], 1)
    assert_array_equal(seg[10:, 10:], 3)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
