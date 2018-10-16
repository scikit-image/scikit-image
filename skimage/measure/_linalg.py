import math

import numpy as np


def distance_point_line(point, line_1, line_2):
    """Return the distance between a point and a line.

    Parameters
    ----------
    point : N-tuple of numeric scalar (float or int)
        The point to find the distance.
    line_1 : N-tuple of numeric scalar (float or int)
        First point the line passes through
    line_2 : N-tuple of numeric scalar (float or int)
        Second point the line passes through

    Returns
    -------
    point : float
        The distance between the point and the line in units

    """
    distance = np.linalg.norm(np.cross(np.subtract(point, line_2),
                                       np.subtract(point, line_1)) /
                              np.linalg.norm(np.subtract(line_2, line_1)))

    return np.abs(distance)


def rotation_matrix(angle, direction, point=None):
    """Returns transformation matrix to rotate about axis defined
    by direction and point.

    Parameters
    ----------
    angle : float
        Angle of rotation (in radians).
    direction : 3-tuple
        The direction of the line to rotate about.
    point : 3-tuple, optional
        A point on the line, if None, default to origin.

    Returns
    -------
    rotation_matrix : (4, 4) array
        The rotation transformation matrix.

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)

    # unit direction vector
    direction = direction / np.linalg.norm(direction)

    # rotation matrix around unit vector
    rot_matrix = np.diag([cosa, cosa, cosa])
    rot_matrix += np.outer(direction, direction) * (1 - cosa)
    direction *= sina
    rot_matrix += np.array([[0, -direction[2], direction[1]],
                            [direction[2], 0, -direction[0]],
                            [-direction[1], direction[0], 0]])

    matrix = np.identity(4)
    matrix[:3, :3] = rot_matrix
    if point is not None:
        # rotation around line from point, add translation element
        matrix[:3, 3] = point - np.dot(rot_matrix, point)
    return matrix


def affine_transform(matrix, points):
    """Affine transform of a set of points by a transformation matrix.

    Parameters
    ----------
    matrix : float
        Angle of rotation (in radians).
    points : list of 3-tuple
        List of 3d points to tranform

    Returns
    -------
    rotation_matrix : (4, 4) array
        The rotation transformation matrix.

    """
    points = np.asarray(points)
    ones = np.ones((points.shape[0], 1))
    points = np.concatenate((points, ones), axis=1)
    return np.dot(points, matrix.T)[..., :3]


def any_perpendicular_vector_3d(vector):
    """Returns a perpendicular vector to the one given, such as
    v1 @ v2 = 0

    Parameters
    ----------
    vector : 3-tuple
        The vector.

    Returns
    -------
    vector : 3-tuple
        The rotation transformation matrix.

    """
    if not np.any(vector):
        raise ValueError('All values are zero')
    if vector[0] == 0 and vector[1] == 0:
        return [0, 1, 0]
    elif vector[0] == 0 and vector[2] == 0:
        return [1, 0, 0]
    elif vector[1] == 0 and vector[2] == 0:
        return [0, 0, 1]

    return [-vector[0], vector[1], 0]
