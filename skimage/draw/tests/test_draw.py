from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from skimage._shared.testing import test_parallel

from skimage.draw import (set_color, line, line_aa, polygon, polygon_perimeter,
                          circle, circle_perimeter, circle_perimeter_aa,
                          ellipse, ellipse_perimeter,
                          _bezier_segment, bezier_curve)


def test_set_color():
    img = np.zeros((10, 10))

    rr, cc = line(0, 0, 0, 30)
    set_color(img, (rr, cc), 1)

    img_ = np.zeros((10, 10))
    img_[0, :] = 1

    assert_array_equal(img, img_)


def test_set_color_with_alpha():
    img = np.zeros((10, 10))

    rr, cc, alpha = line_aa(0, 0, 0, 30)
    set_color(img, (rr, cc), 1, alpha=alpha)

    # Wrong dimensionality color
    assert_raises(ValueError, set_color, img, (rr, cc), (255, 0, 0), alpha=alpha)

    img = np.zeros((10, 10, 3))

    rr, cc, alpha = line_aa(0, 0, 0, 30)
    set_color(img, (rr, cc), (1, 0, 0), alpha=alpha)


@test_parallel()
def test_line_horizontal():
    img = np.zeros((10, 10))

    rr, cc = line(0, 0, 0, 9)
    img[rr, cc] = 1

    img_ = np.zeros((10, 10))
    img_[0, :] = 1

    assert_array_equal(img, img_)


def test_line_vertical():
    img = np.zeros((10, 10))

    rr, cc = line(0, 0, 9, 0)
    img[rr, cc] = 1

    img_ = np.zeros((10, 10))
    img_[:, 0] = 1

    assert_array_equal(img, img_)


def test_line_reverse():
    img = np.zeros((10, 10))

    rr, cc = line(0, 9, 0, 0)
    img[rr, cc] = 1

    img_ = np.zeros((10, 10))
    img_[0, :] = 1

    assert_array_equal(img, img_)


def test_line_diag():
    img = np.zeros((5, 5))

    rr, cc = line(0, 0, 4, 4)
    img[rr, cc] = 1

    img_ = np.eye(5)

    assert_array_equal(img, img_)


def test_line_aa_horizontal():
    img = np.zeros((10, 10))

    rr, cc, val = line_aa(0, 0, 0, 9)
    set_color(img, (rr, cc), 1, alpha=val)

    img_ = np.zeros((10, 10))
    img_[0, :] = 1

    assert_array_equal(img, img_)


def test_line_aa_vertical():
    img = np.zeros((10, 10))

    rr, cc, val = line_aa(0, 0, 9, 0)
    img[rr, cc] = val

    img_ = np.zeros((10, 10))
    img_[:, 0] = 1

    assert_array_equal(img, img_)


def test_line_aa_diagonal():
    img = np.zeros((10, 10))

    rr, cc, val = line_aa(0, 0, 9, 6)
    img[rr, cc] = 1

    # Check that each pixel belonging to line,
    # also belongs to line_aa
    r, c = line(0, 0, 9, 6)
    for x, y in zip(r, c):
        assert_equal(img[r, c], 1)

def test_line_equal_aliasing_horizontally_vertically():
    img0 = np.zeros((25, 25))
    img1 = np.zeros((25, 25))

    # Near-horizontal line
    rr, cc, val = line_aa(10, 2, 12, 20)
    img0[rr, cc] = val

    # Near-vertical (transpose of prior)
    rr, cc, val = line_aa(2, 10, 20, 12)
    img1[rr, cc] = val

    # Difference - should be zero
    assert_array_equal(img0, img1.T)


def test_polygon_rectangle():
    img = np.zeros((10, 10), 'uint8')

    rr, cc = polygon((1, 4, 4, 1, 1), (1, 1, 4, 4, 1))
    img[rr, cc] = 1

    img_ = np.zeros((10, 10))
    img_[1:4, 1:4] = 1

    assert_array_equal(img, img_)


def test_polygon_rectangle_angular():
    img = np.zeros((10, 10), 'uint8')
    poly = np.array(((0, 3), (4, 7), (7, 4), (3, 0), (0, 3)))

    rr, cc = polygon(poly[:, 0], poly[:, 1])
    img[rr, cc] = 1

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


def test_polygon_parallelogram():
    img = np.zeros((10, 10), 'uint8')
    poly = np.array(((1, 1), (5, 1), (7, 6), (3, 6), (1, 1)))

    rr, cc = polygon(poly[:, 0], poly[:, 1])
    img[rr, cc] = 1

    img_ = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    assert_array_equal(img, img_)


def test_polygon_exceed():
    img = np.zeros((10, 10), 'uint8')
    poly = np.array(((1, -1), (100, -1), (100, 100), (1, 100), (1, 1)))

    rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
    img[rr, cc] = 1

    img_ = np.zeros((10, 10))
    img_[1:, :] = 1

    assert_array_equal(img, img_)


def test_circle():
    img = np.zeros((15, 15), 'uint8')

    rr, cc = circle(7, 7, 6)
    img[rr, cc] = 1

    img_ = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    assert_array_equal(img, img_)


def test_circle_perimeter_bresenham():
    img = np.zeros((15, 15), 'uint8')
    rr, cc = circle_perimeter(7, 7, 0, method='bresenham')
    img[rr, cc] = 1
    assert(np.sum(img) == 1)
    assert(img[7][7] == 1)

    img = np.zeros((17, 15), 'uint8')
    rr, cc = circle_perimeter(7, 7, 7, method='bresenham')
    img[rr, cc] = 1
    img_ = np.array(
        [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    assert_array_equal(img, img_)


def test_circle_perimeter_bresenham_shape():
    img = np.zeros((15, 20), 'uint8')
    rr, cc = circle_perimeter(7, 10, 9, method='bresenham', shape=(15, 20))
    img[rr, cc] = 1
    shift = 5
    img_ = np.zeros((15 + 2 * shift, 20), 'uint8')
    rr, cc = circle_perimeter(7 + shift, 10, 9, method='bresenham', shape=None)
    img_[rr, cc] = 1
    assert_array_equal(img, img_[shift:-shift, :])


def test_circle_perimeter_andres():
    img = np.zeros((15, 15), 'uint8')
    rr, cc = circle_perimeter(7, 7, 0, method='andres')
    img[rr, cc] = 1
    assert(np.sum(img) == 1)
    assert(img[7][7] == 1)

    img = np.zeros((17, 15), 'uint8')
    rr, cc = circle_perimeter(7, 7, 7, method='andres')
    img[rr, cc] = 1
    img_ = np.array(
        [[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
         [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
         [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    assert_array_equal(img, img_)


def test_circle_perimeter_aa():
    img = np.zeros((15, 15), 'uint8')
    rr, cc, val = circle_perimeter_aa(7, 7, 0)
    img[rr, cc] = 1
    assert(np.sum(img) == 1)
    assert(img[7][7] == 1)

    img = np.zeros((17, 17), 'uint8')
    rr, cc, val = circle_perimeter_aa(8, 8, 7)
    img[rr, cc] = val * 255
    img_ = np.array(
        [[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
         [  0,   0,   0,   0,   0,  82, 180, 236, 255, 236, 180,  82,   0,   0,   0,   0,   0],
         [  0,   0,   0,   0, 189, 172,  74,  18,   0,  18,  74, 172, 189,   0,   0,   0,   0],
         [  0,   0,   0, 229,  25,   0,   0,   0,   0,   0,   0,   0,  25, 229,   0,   0,   0],
         [  0,   0, 189,  25,   0,   0,   0,   0,   0,   0,   0,   0,   0,  25, 189,   0,   0],
         [  0,  82, 172,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 172,  82,   0],
         [  0, 180,  74,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  74, 180,   0],
         [  0, 236,  18,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  18, 236,   0],
         [  0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0],
         [  0, 236,  18,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  18, 236,   0],
         [  0, 180,  74,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  74, 180,   0],
         [  0,  82, 172,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 172,  82,   0],
         [  0,   0, 189,  25,   0,   0,   0,   0,   0,   0,   0,   0,   0,  25, 189,   0,   0],
         [  0,   0,   0, 229,  25,   0,   0,   0,   0,   0,   0,   0,  25, 229,   0,   0,   0],
         [  0,   0,   0,   0, 189, 172,  74,  18,   0,  18,  74, 172, 189,   0,   0,   0,   0],
         [  0,   0,   0,   0,   0,  82, 180, 236, 255, 236, 180,  82,   0,   0,   0,   0,   0],
         [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]]
    )
    assert_array_equal(img, img_)


def test_circle_perimeter_aa_shape():
    img = np.zeros((15, 20), 'uint8')
    rr, cc, val = circle_perimeter_aa(7, 10, 9, shape=(15, 20))
    img[rr, cc] = val * 255

    shift = 5
    img_ = np.zeros((15 + 2 * shift, 20), 'uint8')
    rr, cc, val = circle_perimeter_aa(7 + shift, 10, 9, shape=None)
    img_[rr, cc] = val * 255
    assert_array_equal(img, img_[shift:-shift, :])


def test_ellipse_trivial():
    img = np.zeros((2, 2), 'uint8')
    rr, cc = ellipse(0.5, 0.5, 0.5, 0.5)
    img[rr, cc] = 1
    img_correct = np.array([
        [0, 0],
        [0, 0]
    ])
    assert_array_equal(img, img_correct)

    img = np.zeros((2, 2), 'uint8')
    rr, cc = ellipse(0.5, 0.5, 1.1, 1.1)
    img[rr, cc] = 1
    img_correct = np.array([
        [1, 1],
        [1, 1],
    ])
    assert_array_equal(img, img_correct)

    img = np.zeros((3, 3), 'uint8')
    rr, cc = ellipse(1, 1, 0.9, 0.9)
    img[rr, cc] = 1
    img_correct = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    assert_array_equal(img, img_correct)

    img = np.zeros((3, 3), 'uint8')
    rr, cc = ellipse(1, 1, 1.1, 1.1)
    img[rr, cc] = 1
    img_correct = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])
    assert_array_equal(img, img_correct)

    img = np.zeros((3, 3), 'uint8')
    rr, cc = ellipse(1, 1, 1.5, 1.5)
    img[rr, cc] = 1
    img_correct = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])
    assert_array_equal(img, img_correct)


def test_ellipse_generic():
    img = np.zeros((4, 4), 'uint8')
    rr, cc = ellipse(1.5, 1.5, 1.1, 1.7)
    img[rr, cc] = 1
    img_ = np.array([
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
    ])
    assert_array_equal(img, img_)

    img = np.zeros((5, 5), 'uint8')
    rr, cc = ellipse(2, 2, 1.7, 1.7)
    img[rr, cc] = 1
    img_ = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    assert_array_equal(img, img_)

    img = np.zeros((10, 10), 'uint8')
    rr, cc = ellipse(5, 5, 3, 4)
    img[rr, cc] = 1
    img_ = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    assert_array_equal(img, img_)

    img = np.zeros((10, 10), 'uint8')
    rr, cc = ellipse(4.5, 5, 3.5, 4)
    img[rr, cc] = 1
    img_ = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    assert_array_equal(img, img_)

    img = np.zeros((15, 15), 'uint8')
    rr, cc = ellipse(7, 7, 3, 7)
    img[rr, cc] = 1
    img_ = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    assert_array_equal(img, img_)


def test_ellipse_with_shape():
    img = np.zeros((15, 15), 'uint8')

    rr, cc = ellipse(7, 7, 3, 10, shape=img.shape)
    img[rr, cc] = 1

    img_ = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    assert_array_equal(img, img_)


def test_ellipse_negative():
    rr, cc = ellipse(-3, -3, 1.7, 1.7)
    rr_, cc_ = np.nonzero(np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]))

    assert_array_equal(rr, rr_ - 5)
    assert_array_equal(cc, cc_ - 5)


def test_ellipse_perimeter_dot_zeroangle():
    # dot, angle == 0
    img = np.zeros((30, 15), 'uint8')
    rr, cc = ellipse_perimeter(15, 7, 0, 0, 0)
    img[rr, cc] = 1
    assert(np.sum(img) == 1)
    assert(img[15][7] == 1)


def test_ellipse_perimeter_dot_nzeroangle():
    # dot, angle != 0
    img = np.zeros((30, 15), 'uint8')
    rr, cc = ellipse_perimeter(15, 7, 0, 0, 1)
    img[rr, cc] = 1
    assert(np.sum(img) == 1)
    assert(img[15][7] == 1)


def test_ellipse_perimeter_flat_zeroangle():
    # flat ellipse
    img = np.zeros((20, 18), 'uint8')
    img_ = np.zeros((20, 18), 'uint8')
    rr, cc = ellipse_perimeter(6, 7, 0, 5, 0)
    img[rr, cc] = 1
    rr, cc = line(6, 2, 6, 12)
    img_[rr, cc] = 1
    assert_array_equal(img, img_)


def test_ellipse_perimeter_zeroangle():
    # angle == 0
    img = np.zeros((30, 15), 'uint8')
    rr, cc = ellipse_perimeter(15, 7, 14, 6, 0)
    img[rr, cc] = 1
    img_ = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]]
    )

    assert_array_equal(img, img_)


def test_ellipse_perimeter_nzeroangle():
    # angle != 0
    img = np.zeros((30, 25), 'uint8')
    rr, cc = ellipse_perimeter(15, 11, 12, 6, 1.1)
    img[rr, cc] = 1
    img_ = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    assert_array_equal(img, img_)


def test_ellipse_perimeter_shape():
    img = np.zeros((15, 20), 'uint8')
    rr, cc = ellipse_perimeter(7, 10, 9, 9, 0, shape=(15, 20))
    img[rr, cc] = 1
    shift = 5
    img_ = np.zeros((15 + 2 * shift, 20), 'uint8')
    rr, cc = ellipse_perimeter(7 + shift, 10, 9, 9, 0, shape=None)
    img_[rr, cc] = 1
    assert_array_equal(img, img_[shift:-shift, :])


def test_bezier_segment_straight():
    image = np.zeros((200, 200), dtype=int)
    x0 = 50
    y0 = 50
    x1 = 150
    y1 = 50
    x2 = 150
    y2 = 150
    rr, cc = _bezier_segment(x0, y0, x1, y1, x2, y2, 0)
    image[rr, cc] = 1

    image2 = np.zeros((200, 200), dtype=int)
    rr, cc = line(x0, y0, x2, y2)
    image2[rr, cc] = 1
    assert_array_equal(image, image2)


def test_bezier_segment_curved():
    img = np.zeros((25, 25), 'uint8')
    x1, y1 = 20, 20
    x2, y2 = 20, 2
    x3, y3 = 2, 2
    rr, cc = _bezier_segment(x1, y1, x2, y2, x3, y3, 1)
    img[rr, cc] = 1
    img_ = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    assert_equal(img[x1, y1], 1)
    assert_equal(img[x3, y3], 1)
    assert_array_equal(img, img_)


def test_bezier_curve_straight():
    image = np.zeros((200, 200), dtype=int)
    x0 = 50
    y0 = 50
    x1 = 150
    y1 = 50
    x2 = 150
    y2 = 150
    rr, cc = bezier_curve(x0, y0, x1, y1, x2, y2, 0)
    image [rr, cc] = 1

    image2 = np.zeros((200, 200), dtype=int)
    rr, cc = line(x0, y0, x2, y2)
    image2 [rr, cc] = 1
    assert_array_equal(image, image2)


def test_bezier_curved_weight_eq_1():
    img = np.zeros((23, 8), 'uint8')
    x1, y1 = (1, 1)
    x2, y2 = (11, 11)
    x3, y3 = (21, 1)
    rr, cc = bezier_curve(x1, y1, x2, y2, x3, y3, 1)
    img[rr, cc] = 1
    assert_equal(img[x1, y1], 1)
    assert_equal(img[x3, y3], 1)
    img_ = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]]
    )
    assert_equal(img, img_)


def test_bezier_curved_weight_neq_1():
    img = np.zeros((23, 10), 'uint8')
    x1, y1 = (1, 1)
    x2, y2 = (11, 11)
    x3, y3 = (21, 1)
    rr, cc = bezier_curve(x1, y1, x2, y2, x3, y3, 2)
    img[rr, cc] = 1
    assert_equal(img[x1, y1], 1)
    assert_equal(img[x3, y3], 1)
    img_ = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    assert_equal(img, img_)


def test_bezier_curve_shape():
    img = np.zeros((15, 20), 'uint8')
    x1, y1 = (1, 5)
    x2, y2 = (6, 11)
    x3, y3 = (1, 14)
    rr, cc = bezier_curve(x1, y1, x2, y2, x3, y3, 2, shape=(15, 20))
    img[rr, cc] = 1
    shift = 5
    img_ = np.zeros((15 + 2 * shift, 20), 'uint8')
    x1, y1 = (1 + shift, 5)
    x2, y2 = (6 + shift, 11)
    x3, y3 = (1 + shift, 14)
    rr, cc = bezier_curve(x1, y1, x2, y2, x3, y3, 2, shape=None)
    img_[rr, cc] = 1
    assert_array_equal(img, img_[shift:-shift, :])


def test_polygon_perimeter():
    expected = np.array(
        [[1, 1, 1, 1],
         [1, 0, 0, 1],
         [1, 1, 1, 1]]
    )
    out = np.zeros_like(expected)

    rr, cc = polygon_perimeter([0, 2, 2, 0],
                               [0, 0, 3, 3])

    out[rr, cc] = 1
    assert_array_equal(out, expected)

    out = np.zeros_like(expected)
    rr, cc = polygon_perimeter([-1, -1, 3,  3],
                               [-1,  4, 4, -1],
                               shape=out.shape, clip=True)
    out[rr, cc] = 1
    assert_array_equal(out, expected)

    assert_raises(ValueError, polygon_perimeter, [0], [1], clip=True)


def test_polygon_perimeter_outside_image():
    rr, cc = polygon_perimeter([-1, -1, 3,  3],
                               [-1,  4, 4, -1], shape=(3, 4))
    assert_equal(len(rr), 0)
    assert_equal(len(cc), 0)


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
