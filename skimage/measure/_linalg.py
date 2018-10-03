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
                                       np.subtract(point, line_1)) / np.linalg.norm(np.subtract(line_2, line_1)))

    return np.abs(distance)


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    #direction = unit_vector(direction[:3])
    # unit direction vector
    direction = direction[:3] / np.linalg.norm(direction[:3])

    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                  [ direction[2], 0.0,          -direction[0]],
                  [-direction[1], direction[0],  0.0]])

    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def get_any_perpendicular_vector_3d(v):
    """

    :param v:
    :return:
    """
    if not np.any(v):
        raise ValueError('All values are zero')
    if v[0] == 0 and v[1] == 0:
        return [0, 1, 0]
    elif v[0] == 0 and v[2] == 0:
        return [1, 0, 0]
    elif v[1] == 0 and v[2] == 0:
        return [0, 0, 1]

    return [-v[0], v[1], 0]
