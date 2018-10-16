import numpy as np
from scipy import constants

from skimage.measure._linalg import distance_point_line, \
    any_perpendicular_vector_3d, rotation_matrix, affine_transform
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


def test_rotate_point_around_line_45():
    points = np.asarray([[1, 0, 0]])
    direction = [0, 0, 1]
    rot_matrix = rotation_matrix(constants.pi / 4, direction)
    transformed_points = affine_transform(rot_matrix, points)
    assert_almost_equal(transformed_points[0],
                        [np.sqrt(2) / 2, np.sqrt(2) / 2, 0])


def test_rotate_point_around_line_90():
    points = np.asarray([[1, 0, 0]])
    direction = [0, 0, 1]
    rot_matrix = rotation_matrix(constants.pi / 2, direction)
    transformed_points = affine_transform(rot_matrix, points)
    assert_almost_equal(transformed_points[0], [0, -1, 0])


def test_rotate_point_around_line_180():
    points = np.asarray([[1, 0, 0]])
    direction = [0, 0, 1]
    rot_matrix = rotation_matrix(constants.pi, direction)
    transformed_points = affine_transform(rot_matrix, points)
    assert_almost_equal(transformed_points[0], [-1, 0, 0])


def test_rotate_point_around_line_180_offset():
    points = np.asarray([[1, 0, 0]])
    direction = [1, 0, 0]
    point_on_line = [0, 0, 1]
    rot_matrix = rotation_matrix(constants.pi, direction, point_on_line)
    transformed_points = affine_transform(rot_matrix, points)
    assert_almost_equal(transformed_points[0], [1, 0, 2])


def test_rotate_point_around_diag_line_180():
    points = np.asarray([[1, 0, 0]])
    direction = [1, 1, 1]
    rot_matrix = rotation_matrix(constants.pi, direction)
    transformed_points = affine_transform(rot_matrix, points)
    assert_almost_equal(transformed_points[0], [-1/3, 2/3, 2/3])


def test_rotate_point_around_diag_line_minus_180():
    points = np.asarray([[1, 0, 0]])
    direction = [1, 1, 1]
    rot_matrix = rotation_matrix(-constants.pi, direction)
    transformed_points = affine_transform(rot_matrix, points)
    assert_almost_equal(transformed_points[0], [-1/3, 2/3, 2/3])


def test_rotate_point_on_line():
    points = np.asarray([[1, 1, 1], [0, 0, 0]])
    direction = [1, 0, 0]
    rot_matrix = rotation_matrix(constants.pi, direction, points[0])
    transformed_points = affine_transform(rot_matrix, points)
    assert_almost_equal(transformed_points, points)


def test_affine_transform():
    matrix = np.identity(4)
    points = [[1, 1, 1], [2, 2, 2]]
    transformed_points = affine_transform(matrix, points)
    assert_equal(transformed_points, points)


def test_get_any_perpendicular_vector():
    v1 = [1, 1, 1]
    v2 = any_perpendicular_vector_3d(v1)
    assert_equal(np.dot(v1, v2), 0)
