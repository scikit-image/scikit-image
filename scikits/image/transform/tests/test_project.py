import numpy as np
from numpy.testing import assert_array_almost_equal

from scikits.image.transform.project import _stackcopy, homography

def test_stackcopy():
    layers = 4
    x = np.empty((3, 3, layers))
    y = np.eye(3, 3)
    _stackcopy(x, y)
    for i in range(layers):
        assert_array_almost_equal(x[...,i], y)

def test_homography():
    x = np.arange(9).reshape((3, 3)) + 1
    theta = -np.pi/2
    M = np.array([[np.cos(theta),-np.sin(theta),0],
                  [np.sin(theta), np.cos(theta),2],
                  [0,             0,            1]])
    x90 = homography(x, M, order=1)
    assert_array_almost_equal(x90, np.rot90(x))
