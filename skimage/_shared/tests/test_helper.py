import numpy as np
from numpy.testing import assert_array_equal, assert_raises
from skimage._shared.helper import ensure_gray


def test_ensure_gray():
    image = np.zeros((10, 10))
    out = ensure_gray(image, convert=True)
    assert_array_equal(image, out)

    image = np.zeros((10, 10, 3))
    out = ensure_gray(image, convert=True)
    assert_array_equal((10, 10), out.shape)
    assert_raises(ValueError, ensure_gray, image, convert=False)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
