import numpy as np

from skimage import img_as_float
from skimage.feature import descriptor_shapecontext


def test_square():
    I = np.zeros((50, 50)).astype(float)
    I[:25, :25] = 1.
    x = (10, 10)
    results = descriptor_shapecontext(I, 0, 20, x, radial_bins=2, polar_bins=6)
    assert results.any()
    assert results.shape == (2, 6)


def test_squared_dot():
    I = np.zeros((20, 20))
    I[4:8, 4:8] = 1
    I = img_as_float(I)
    x = (10, 10)
    results = descriptor_shapecontext(I, 0, 25, x, radial_bins=3, polar_bins=4)
    expected = np.zeros((3, 4))
    expected[1, 2] = 16
    assert (results.astype(int) == expected).all()
