import numpy as np
from scipy import constants


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


def rotation_angles_by_distance_from_line(dst, src, point):
    """Return an array of angles that will be used to rotate a point 360 degrees around a line.
    The number of angles is dependent on the distance between the point and the line.
    The farther the point from the line, the smaller the angle incrementation.

    Parameters
    ----------
    src : 3-tuple of numeric scalar (float or int)
        A first point where the line is passing through
    dst : 3-tuple of numeric scalar (float or int)
        A second point where the line is passing through
    point : 3-tuple of numeric scalar (float or int)
        The point to find the distance.

    Returns
    -------
    angles : tuple, float
        The angles that will be used to rotate the sample point to create more sample points around the line.
    """
    dst = distance_point_line(point, src, dst)
    if dst == 0:
        rotation_angles = np.zeros(1)
    else:
        rotation_angles = np.linspace(0, 2 * constants.pi, 2 * dst + 3)  # todo, what is the 3??
        rotation_angles = rotation_angles[:-1]
    return rotation_angles


def rotate_sample_points(perp_array, src, dst):
    """Return the evenly rotated coordinates of the sample points along a scan line in 3d

    Parameters
    ----------
    perp_array, shape (3, M, N), float
        The coordinates of the profile along the scan line. The length of the
        profile is the ceil of the computed length of the scan line.
        The coordinates are 180 degrees apart.
    src : 3-tuple of numeric scalar (float or int)
        A first point where the line is passing through
    dst : 3-tuple of numeric scalar (float or int)
        A second point where the line is passing through

    Returns
    -------
    sampling_array : array, shape (3, M, N), float
        The coordinates of the 3d sample points along the scan line. The length of the
        profile is the ceil of the computed length of the scan line.
    """
    line_vector = np.subtract(dst, src)
    length_vector = np.linalg.norm(line_vector)
    unit_direction_vector = np.divide(line_vector, length_vector)

    # Rotate the points around the axis a number of times depending on the distance of the point
    # from the direction axis to simulate sampling of points around the axis
    sampling_array = []
    for perp_points in np.transpose(perp_array):
        rotation_angles = rotation_angles_by_distance_from_line(dst, src, perp_points[0])
        for angle in rotation_angles:  # the number of angles to use as rotation angles for the samping points
            points_array = []
            for point in perp_points:  # the number of unit points on displacement vector
                if angle == 0:
                    points_array.append(point)
                else:
                    rotated_point = rotate_point_around_line(point, src, unit_direction_vector, angle)
                    points_array.append(rotated_point)

            sampling_array.append(points_array)

    # Return transposed array for ndi.map_coordinates
    return np.transpose(np.array(sampling_array, dtype=float))
