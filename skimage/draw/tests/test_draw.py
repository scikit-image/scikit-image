from numpy.testing import assert_array_equal
import numpy as np

from skimage.draw import bresenham

def test_bresenham_horizontal():
    img = np.zeros((10, 10))

    rr, cc = bresenham(0, 0, 0, 9)
    img[rr, cc] = 1

    img_ = np.zeros((10, 10))
    img_[0, :] = 1

    assert_array_equal(img, img_)

def test_bresenham_vertical():
    img = np.zeros((10, 10))

    rr, cc = bresenham(0, 0, 9, 0)
    img[rr, cc] = 1

    img_ = np.zeros((10, 10))
    img_[:, 0] = 1

    assert_array_equal(img, img_)

def test_reverse():
    img = np.zeros((10, 10))

    rr, cc = bresenham(0, 9, 0, 0)
    img[rr, cc] = 1

    img_ = np.zeros((10, 10))
    img_[0, :] = 1

    assert_array_equal(img, img_)

def test_diag():
    img = np.zeros((5, 5))

    rr, cc = bresenham(0, 0, 4, 4)
    img[rr, cc] = 1

    img_ = np.eye(5)

    assert_array_equal(img, img_)


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    
