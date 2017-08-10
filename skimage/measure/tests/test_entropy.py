from numpy.testing import assert_almost_equal
import numpy as np

from skimage.measure import shannon_entropy


def test_shannon_ones():
    img = np.ones((10, 10))
    res = shannon_entropy(img, base=np.e)
    assert_almost_equal(res, 0.0)

def test_shannon_all_unique():
    img = np.arange(64)
    res = shannon_entropy(img, base=2)
    assert_almost_equal(res, np.log(64)/np.log(2))

if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
