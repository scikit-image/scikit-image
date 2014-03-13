import numpy as np

from skimage import img_as_float
from skimage.feature import shapecontext


def test_square():
    im = np.zeros((50, 50)).astype(float)
    im[:25, :25] = 1.
    x = (10, 10)
    results = shapecontext(im, 0, 20, x, radial_bins=2, polar_bins=6)
    assert results.any()
    assert results.shape == (2, 6)


def test_squared_dot():
    im = np.zeros((20, 20))
    im[4:8, 4:8] = 1
    im = img_as_float(im)
    x = (10, 10)
    results = shapecontext(im, 0, 25, x, radial_bins=3, polar_bins=4)
    expected = np.zeros((3, 4))
    expected[1, 2] = 16
    assert (results.astype(int) == expected).all()


if __name__ == '__main__':
    from numpy.testing import run_module_suite
    run_module_suite()
