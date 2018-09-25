import numpy as np
from scipy import constants
from skimage.measure._linalg import distance_point_line, rotate_point_around_line, rotation_angles_by_distance_from_line
from skimage._shared.testing import assert_equal, assert_almost_equal


def test_distance_point_line_on_line():
    point = [5, 5]
    line_1 = [0, 0]
    line_2 = [10, 10]
    distance = distance_point_line(point, line_1, line_2)
    assert_equal(distance, 0)


def test_distance_point_line_vertical():
    point = [5, 5]
    line_1 = [0, 0]
    line_2 = [0, 10]
    distance = distance_point_line(point, line_1, line_2)
    assert_equal(distance, 5)


def test_distance_point_line_diagonal():
    point = [1, 3]
    line_1 = [0, 0]
    line_2 = [10, 10]
    distance = distance_point_line(point, line_1, line_2)
    assert_almost_equal(distance, np.sqrt(2))


def test_distance_point_line_on_line_3d():
    point = [0, 0, 0]
    line_1 = [0, 0, 0]
    line_2 = [10, 10, 10]
    distance = distance_point_line(point, line_1, line_2)
    assert_equal(distance, 0)


def test_distance_point_line_vertical_3d():
    point = [0, 5, 5]
    line_1 = [0, 0, 0]
    line_2 = [0, 0, 10]
    distance = distance_point_line(point, line_1, line_2)
    assert_equal(distance, 5)


def test_distance_point_line_diagonal_3d():
    point = [1, 2, 3]
    line_1 = [0, 0, 0]
    line_2 = [10, 10, 10]
    distance = distance_point_line(point, line_1, line_2)
    assert_almost_equal(distance, np.sqrt(2))


def test_rotate_point_around_line_90():
    point = [1, 0, 0]
    line = [0, 0, 0]
    direction = [0, 0, 1]
    rotated_point = rotate_point_around_line(point, line, direction, constants.pi/2)
    assert_almost_equal(rotated_point, [0, -1, 0])


def test_rotate_point_around_line_180():
    point = [1, 0, 0]
    line = [0, 0, 0]
    direction = [0, 0, 1]
    rotated_point = rotate_point_around_line(point, line, direction, constants.pi)
    assert_almost_equal(rotated_point, [-1, 0, 0])


def test_rotate_point_around_diag_line_180():
    point = [1, 0, 0]
    line = [0, 0, 0]
    direction = [1, 1, 1]
    rotated_point = rotate_point_around_line(point, line, direction, constants.pi)
    assert_almost_equal(rotated_point, [1, 2, 2])


def test_rotate_point_on_line():
    point = [1, 1, 1]
    line = [0, 0, 0]
    direction = [1, 1, 1]
    rotated_point = rotate_point_around_line(point, line, direction, constants.pi/5)
    assert_almost_equal(rotated_point, point)


def test_rotate_sample_points():

    pass

def test_rotation_angles_by_distance_from_line():
    rotation_angles_by_distance_from_line()
    pass