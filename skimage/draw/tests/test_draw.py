from numpy.testing import assert_array_equal
import numpy as np

from skimage.draw import bresenham, fill_polygon


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

def test_bresenham_reverse():
    img = np.zeros((10, 10))

    rr, cc = bresenham(0, 9, 0, 0)
    img[rr, cc] = 1

    img_ = np.zeros((10, 10))
    img_[0, :] = 1

    assert_array_equal(img, img_)

def test_bresenham_diag():
    img = np.zeros((5, 5))

    rr, cc = bresenham(0, 0, 4, 4)
    img[rr, cc] = 1

    img_ = np.eye(5)

    assert_array_equal(img, img_)



def test_fill_polygon_rectangle():
    img = np.zeros((10, 10), 'uint8')
    poly = np.array(((1, 1), (4, 1), (4, 4), (1, 4), (1, 1)))

    fill_polygon(img, poly)

    img_ = np.zeros((10, 10))
    img_[2:5,1:4] = 1

    assert_array_equal(img, img_)

def test_fill_polygon_rectangle_angular():
    img = np.zeros((10, 10), 'uint8')
    poly = np.array(((0, 3), (4, 7), (7, 4), (3, 0), (0, 3)))

    fill_polygon(img, poly)

    img_ = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    assert_array_equal(img, img_)

def test_fill_polygon_parallelogram():
    img = np.zeros((10, 10), 'uint8')
    poly = np.array(((1, 1), (5, 1), (7, 6), (3, 6), (1, 1)))

    fill_polygon(img, poly)

    print img

    img_ = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    assert_array_equal(img, img_)

def test_fill_polygon_color():
    img = np.zeros((10, 10), 'uint8')
    poly = np.array(((1, 1), (4, 1), (4, 4), (1, 4), (1, 1)))

    fill_polygon(img, poly, color=123)

    img_ = np.zeros((10, 10))
    img_[2:5,1:4] = 123

    assert_array_equal(img, img_)


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
