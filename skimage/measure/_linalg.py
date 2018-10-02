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


def rotation_matrix(axis, theta):
    from scipy.linalg import expm

    # def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / np.linalg.norm(axis) * theta))


def rotate_point_around_line(point_to_rotate, point_on_line, unit_direction_vector, angle_in_radians):
    """Return a 3d point that is rotated at an angle of theta around a line
    passing through a selected point.

    Parameters
    ----------
    point_to_rotate : 3-tuple of numeric scalar (float or int)
        The point to rotate.
    point_on_line : 3-tuple of numeric scalar (float or int)
        A point where the line is passing through
    unit_direction_vector : 3-tuple of numeric scalar (float or int)
        The unit direction vector of the line
    angle_in_radians : float or int
        The angle of rotation in radians

    Returns
    -------
    return_value : array, shape (3), float
        The coordinates of the rotated point around the line
    """
    c, b, a = point_on_line
    r, q, p = point_to_rotate
    w, v, u = unit_direction_vector

    p1 = (a * (v ** 2 + w ** 2) - u * (b * v + c * w - u * p - v * q - w * r)) * \
         (1 - np.cos(angle_in_radians)) + p * np.cos(angle_in_radians) + \
         (-c * v + b * w - w * q + v * r) * np.sin(angle_in_radians)

    p2 = (b * (u ** 2 + w ** 2) - v * (a * u + c * w - u * p - v * q - w * r)) * \
         (1 - np.cos(angle_in_radians)) + q * np.cos(angle_in_radians) + \
         (-c * u - a * w + w * p - u * r) * np.sin(angle_in_radians)

    p3 = (c * (u ** 2 + v ** 2) - w * (a * u + b * v - u * p - v * q - w * r)) * \
         (1 - np.cos(angle_in_radians)) + r * np.cos(angle_in_radians) + \
         (-b * u + a * v - v * p + u * q) * np.sin(angle_in_radians)

    from scipy.linalg import expm

    def M(axis, theta):
        return expm(np.cross(np.eye(3), axis / np.linalg.norm(axis) * theta))

    v, axis, theta = [3, 5, 0], [4, 4, 1], 1.2
    M0 = M(axis, theta)
    x = np.dot(M0, v)
    #print(np.dot(M0, v))

    return np.array([p3, p2, p1])


def get_any_perpendicular_vector(v):
    if not np.any(v):
        raise ValueError('All values are zero')
    if v[0] == 0 and v[1] == 0:
        return [0, 1, 0]
    elif v[0] == 0 and v[2] == 0:
        return [1, 0, 0]
    elif v[1] == 0 and v[2] == 0:
        return [0, 0, 1]

    return [-v[0], v[1], 0]
