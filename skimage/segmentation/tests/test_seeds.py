import itertools as it
import numpy as np
from numpy.testing import assert_equal, assert_raises
from skimage.segmentation import seeds
from skimage._shared.testing import test_parallel


@test_parallel()
def test_color_2d():
    rnd = np.random.RandomState(0)
    img = np.zeros((32, 33, 3))
    img[:16, :16, 0] = 1
    img[16:, :16, 1] = 1
    img[16:, 16:, 2] = 1
    img = img * 0.99 + 0.01 * rnd.uniform(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    # seg = seeds(img, hist_size=8, num_superpixels=4)
    labels = seeds(img, hist_size=12, num_superpixels=4, n_levels=2)

    # we expect 4 segments
    assert_equal(len(np.unique(labels)), 4)
    assert_equal(labels.shape, img.shape[:-1])
    assert_equal(labels[:16, :16], 1)
    assert_equal(labels[16:, :16], 3)
    assert_equal(labels[:16, 16:], 2)
    assert_equal(labels[16:, 16:], 4)

if __name__ == '__main__':
    from numpy import testing

    testing.run_module_suite()
