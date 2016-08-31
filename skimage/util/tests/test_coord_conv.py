import numpy as np
from numpy.testing import assert_array_equal
from skimage.util import (xy_to_rc, rc_to_xy,
                          cart_to_rc, rc_to_cart,
                          cart_to_xy, xy_to_cart)


def test_round_trip_xy_via_cart():
    shape = (5, 10)
    points = np.stack((np.random.randint(0, shape[0], 5),
                       np.random.randint(0, shape[1], 5))).T
    image = np.random.random(shape)
    tmp_p, tmp_i = xy_to_cart(points=points, image=image)
    result_points, result_image = cart_to_xy(points=tmp_p, image=tmp_i)
    assert_array_equal(points, result_points)
    assert_array_equal(image, result_image)


def test_round_trip_xy_via_rc():
    shape = (10, 5)
    points = np.stack((np.random.randint(0, shape[0], 5),
                       np.random.randint(0, shape[1], 5))).T
    image = np.random.random(shape)
    result_points, result_image = rc_to_xy(*xy_to_rc(points, image))
    assert_array_equal(points, result_points)
    assert_array_equal(image, result_image)


def test_round_trip_rc_via_cart():
    shape = (5, 10)
    points = np.stack((np.random.randint(0, shape[0], 5),
                       np.random.randint(0, shape[1], 5))).T
    image = np.random.random(shape)
    tmp_p, tmp_i = rc_to_cart(points=points, image=image)
    result_points, result_image = cart_to_rc(points=tmp_p, image=tmp_i)
    assert_array_equal(points, result_points)
    assert_array_equal(image, result_image)


def test_round_trip_rc_via_xy():
    shape = (10, 5)
    points = np.stack((np.random.randint(0, shape[0], 5),
                       np.random.randint(0, shape[1], 5))).T
    image = np.random.random(shape)
    result_points, result_image = xy_to_rc(*rc_to_xy(points, image))
    assert_array_equal(points, result_points)
    assert_array_equal(image, result_image)


def test_round_trip_cart_via_rc():
    shape = (5, 10)
    points = np.stack((np.random.randint(0, shape[0], 5),
                       np.random.randint(0, shape[1], 5))).T
    image = np.random.random(shape)
    tmp_p, tmp_i = cart_to_rc(points=points, image=image)
    result_points, result_image = rc_to_cart(points=tmp_p, image=tmp_i)
    assert_array_equal(points, result_points)
    assert_array_equal(image, result_image)


def test_round_trip_cart_via_xy():
    shape = (10, 5)
    points = np.stack((np.random.randint(0, shape[0], 5),
                       np.random.randint(0, shape[1], 5))).T
    image = np.random.random(shape)
    tmp_p, tmp_i = cart_to_xy(points=points, image=image)
    result_points, result_image = xy_to_cart(points=tmp_p, image=tmp_i)
    assert_array_equal(points, result_points)
    assert_array_equal(image, result_image)


if __name__ == '__main__':
    np.testing.run_module_suite()
