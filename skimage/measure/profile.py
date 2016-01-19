import numpy as np
from scipy import ndimage as ndi, constants
from ..util import img_as_float


def profile_line(img, src, dst, linewidth=1,
                 order=1, mode='constant', cval=0.0, multichannel=True):
    """Return the intensity profile of an image measured along a scan line.

    Parameters
    ----------
    img : numeric array, shape (M, N[, C])
        The image, either grayscale (2D array) or multichannel
        (3D array, where the final axis contains the channel
        information).
    src : 2-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 2-tuple of numeric scalar (float or int)
        The end point of the scan line.
    linewidth : int, optional
        Width of the scan, perpendicular to the line
    order : int in {0, 1, 2, 3, 4, 5}, optional
        The order of the spline interpolation to compute image values at
        non-integer coordinates. 0 means nearest-neighbor interpolation.
    mode : {'constant', 'nearest', 'reflect', 'mirror', 'wrap'}, optional
        How to compute any values falling outside of the image.
    cval : float, optional
        If `mode` is 'constant', what constant value to use outside the image.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.
    Returns
    -------
    return_value : array
        The intensity profile along the scan line. The length of the profile
        is the ceil of the computed length of the scan line.

    Examples
    --------
    >>> x = np.array([[1, 1, 1, 2, 2, 2]])
    >>> img = np.vstack([np.zeros_like(x), x, x, x, np.zeros_like(x)])
    >>> img
    array([[0, 0, 0, 0, 0, 0],
           [1, 1, 1, 2, 2, 2],
           [1, 1, 1, 2, 2, 2],
           [1, 1, 1, 2, 2, 2],
           [0, 0, 0, 0, 0, 0]])
    >>> profile_line(img, (2, 1), (2, 4))
    array([ 1.,  1.,  2.,  2.])

    Notes
    -----
    The destination point is included in the profile, in contrast to
    standard numpy indexing.
    """

    if img.ndim == 4 and multichannel:
        # 3D RGB to be implemented
        raise NotImplementedError('3D RGB to be implemented')
    elif img.ndim == 3 and not multichannel:
        # 3D intensity
        perp_lines = _line_profile_coordinates3d(src, dst, linewidth=linewidth)
        # Convert 3d array to float, otherwise the order value in ndi.map_coordinates
        # will not be evaluated properly for int arrays
        img = img.astype(np.float)
        pixels = ndi.map_coordinates(img, perp_lines, order=order, mode=mode, cval=cval)
    elif img.ndim == 3 and multichannel:
        # 2D RGB
        perp_lines = _line_profile_coordinates(src, dst, linewidth=linewidth)
        # Convert image to float, otherwise if the image is int, the order value in ndi.map_coordinates
        # will not be evaluated properly
        img = img_as_float(img)
        pixels = [ndi.map_coordinates(img[..., i], perp_lines, order=order,
                                      mode=mode, cval=cval) for i in range(img.shape[2])]
        pixels = np.transpose(np.asarray(pixels), (1, 2, 0))
    elif img.ndim == 2:
        # 2D intensity
        perp_lines = _line_profile_coordinates(src, dst, linewidth=linewidth)
        img = img_as_float(img)
        pixels = ndi.map_coordinates(img, perp_lines, order=order, mode=mode, cval=cval)
    else:
        raise ValueError('Invalid arguments')

    intensities = pixels.mean(axis=1)
    return intensities


def _line_profile_coordinates(src, dst, linewidth=1):
    """Return the coordinates of the profile of an image along a scan line.

    Parameters
    ----------
    src : 2-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 2-tuple of numeric scalar (float or int)
        The end point of the scan line.
    linewidth : int, optional
        Width of the scan, perpendicular to the line

    Returns
    -------
    coords : array, shape (2, N, C), float
        The coordinates of the profile along the scan line. The length of the
        profile is the ceil of the computed length of the scan line.

    Notes
    -----
    This is a utility method meant to be used internally by skimage functions.
    The destination point is included in the profile, in contrast to
    standard numpy indexing.
    """
    src_row, src_col = src = np.asarray(src, dtype=float)
    dst_row, dst_col = dst = np.asarray(dst, dtype=float)
    d_row, d_col = dst - src
    theta = np.arctan2(d_row, d_col)

    length = np.ceil(np.hypot(d_row, d_col) + 1)
    # we add one above because we include the last point in the profile
    # (in contrast to standard numpy indexing)
    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)

    # we subtract 1 from linewidth to change from pixel-counting
    # (make this line 3 pixels wide) to point distances (the
    # distance between pixel centers)
    col_width = (linewidth - 1) * np.sin(-theta) / 2
    row_width = (linewidth - 1) * np.cos(theta) / 2
    perp_rows = np.array([np.linspace(row_i - row_width, row_i + row_width,
                                      linewidth) for row_i in line_row])
    perp_cols = np.array([np.linspace(col_i - col_width, col_i + col_width,
                                      linewidth) for col_i in line_col])
    return np.array([perp_rows, perp_cols])


def _line_profile_coordinates3d(src, dst, linewidth=1):
    """Return the coordinates of the profile of an image along a scan line.

    Parameters
    ----------
    src : 3-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 3-tuple of numeric scalar (float or int)
        The end point of the scan line.
    linewidth : int, optional
        Width of the scan, perpendicular to the line,
        since the line is 3d, this value is the diameter
        of a 3d cylinder following the scan line.

    Returns
    -------
    coords : array, shape (3, N, C), float
        The coordinates of the profile along the scan line. The length of the
        profile is the ceil of the computed length of the scan line.

    Notes
    -----
    This is a utility method meant to be used internally by skimage functions.
    The destination point is included in the profile, in contrast to
    standard numpy indexing.
    """
    src_row, src_col, src_dim3 = src = np.asarray(src, dtype=float)
    dst_row, dst_col, dst_dim3 = dst = np.asarray(dst, dtype=float)
    d_row, d_col, d_plane = dst - src

    # Get one unit vector perpendicular to direction vector to find a point
    # that is one unit distance away from the destination vector
    # (ex: ix + jy + kz = 0, then we can use x = y = 1)
    # Try with z first if it is not 0, then the same for x, otherwise pick y
    # We subtract 1 from linewidth to change from pixel-counting
    # (make this line 3 pixels wide) to point distances (the
    # distance between pixel centers)
    if d_plane != 0:
        # try finding the solution to ix + jy + kz = 0 for x = 1 and y = 1
        dim_z = - (d_row + d_col) / d_plane
        length_vector = np.sqrt(2 + dim_z ** 2)
        col_width = row_width = (linewidth - 1) / (2 * length_vector)
        slice_width = (linewidth - 1) * (dim_z / 2 * length_vector)
    elif d_row != 0:
        # try finding the solution to ix + jy + kz = 0 for y = 1 and z = 1
        dim_x = - (d_plane + d_col) / d_row
        length_vector = np.sqrt(2 + dim_x ** 2)
        col_width = slice_width = (linewidth - 1) / (2 * length_vector)
        row_width = (linewidth - 1) * (dim_x / length_vector) / 2
    else:
        # try finding the solution to ix + jy + kz = 0 for x = 1 and z = 1
        dim_y = - (d_row + d_plane) / d_col
        length_vector = np.sqrt(2 + dim_y ** 2)
        row_width = slice_width = (linewidth - 1) / (2 * length_vector)
        col_width = (linewidth - 1) * (dim_y / length_vector) / 2

    # we add one above because we include the last point in the profile
    # (in contrast to standard numpy indexing)
    length = np.ceil(np.linalg.norm([d_row, d_col, d_plane]) + 1)

    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)
    line_slice = np.linspace(src_dim3, dst_dim3, length)

    perp_rows = np.array([np.linspace(row_i - row_width, row_i + row_width,
                                      linewidth) for row_i in line_row])
    perp_cols = np.array([np.linspace(col_i - col_width, col_i + col_width,
                                      linewidth) for col_i in line_col])
    perp_plane = np.array([np.linspace(slice_i - slice_width, slice_i + slice_width,
                                       linewidth) for slice_i in line_slice])

    perp_array = np.array([perp_rows, perp_cols, perp_plane])

    # rotate all sample points around the direction axis if linewidth is > 1
    if linewidth > 1:
        perp_array = rotate_sample_points(linewidth, perp_array, src, dst)

    return perp_array


def rotate_sample_points(linewidth, perp_array, src, dst):
    """Return the evenly rotated coordinates of the sample points along a scan line in 3d

    Parameters
    ----------
    linewidth : int
        Width of the scan, perpendicular to the line,
        since the line is 3d, this value is the diameter
        of a 3d cylinder following the scan line.
    perp_array, shape (3, N, C), float
        The coordinates of the profile along the scan line. The length of the
        profile is the ceil of the computed length of the scan line.
        The coordinates are 180 degrees apart.
    src : 3-tuple of numeric scalar (float or int)
        A first point where the line is passing through
    dst : 3-tuple of numeric scalar (float or int)
        A second point where the line is passing through

    Returns
    -------
    sampling_array : array, shape (3, N, C), float
        The coordinates of the 3d sample points along the scan line. The length of the
        profile is the ceil of the computed length of the scan line.

    """

    d_row, d_col, d_plane = dst - src
    length = np.ceil(np.linalg.norm([d_row, d_col, d_plane]) + 1)
    unit_direction_vector = [d_row / (length - 1), d_col / (length - 1), d_plane / (length - 1)]

    # Rotate the points around the axis a number of times depending on the linewidth
    # to simulate sampling of points around the axis
    # Example, for a linewidth of 2 only one rotation of 90 degrees for is necessary to get the sampling points
    # For a linewidth of 3 only one rotation of 45 - 90 and 135 degrees are necessary to get the sampling points
    # The center point is not be rotated as it is unnecessary

    rotation_angles = np.linspace(0, constants.pi, (linewidth * 2) - 1)  #
    # Remove last element, the 180 degree rotation, from the list
    rotation_angles = np.delete(rotation_angles, len(rotation_angles) - 1)

    # loop through all the angles to get the rotated sampling points
    sampling_array = []
    for perp_points in perp_array.T: # the number of sample points per unit (i.e. linewidth)
        for angle in rotation_angles:  # the number of angles to use as rotation angles for the samping points
            points_array = []
            for point in perp_points: # the number of unit points on displacement vector
                if angle == 0:
                    points_array.append(point)
                else:
                    # Check to see if point is on axis (i.e. when linewidth is odd)
                    # and ignore, since it has already been added for rotation = 0
                    if _distance_point_line_3d(point, src, dst) == 0:
                        continue
                    rotated_point = np.array(_rotate_point_around_line(point, src, unit_direction_vector, angle))
                    points_array.append(rotated_point)
            # Prevent empty arrays from being added, when a center point is detected for example
            if len(points_array) > 0:
                sampling_array.append(points_array)

    # Return transposed array for ndi.map_coordinates
    return np.array(sampling_array, dtype=float).T


def _distance_point_line_3d(point, src, dst):
    """Return the distance between a line and a point in 3d.

    Parameters
    ----------
    point : 3-tuple of numeric scalar (float or int)
        The point to find the distance.
    src : 3-tuple of numeric scalar (float or int)
        A first point where the line is passing through
    dst : 3-tuple of numeric scalar (float or int)
        A second point where the line is passing through

    Returns
    -------
    point : float
        The distance between the point and the line in units

    """

    line_vector = np.subtract(dst, src)
    length_line = np.linalg.norm(line_vector)
    vector1 = np.subtract(point, dst)
    vector2 = np.subtract(point, src)
    distance = np.linalg.norm(np.cross(vector1, vector2) / length_line)
    return np.abs(distance)


def _rotate_point_around_line(point_to_rotate, point_on_line, unit_direction_vector, angle_in_radians):
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
    unit_direction_vector : float or int
        The angle of rotation in radians

    Returns
    -------
    point : array, shape (3), float
        The coordinates of the rotated point around the line

    """

    a = point_on_line[0]
    b = point_on_line[1]
    c = point_on_line[2]

    x = point_to_rotate[0]
    y = point_to_rotate[1]
    z = point_to_rotate[2]

    u = unit_direction_vector[0]
    v = unit_direction_vector[1]
    w = unit_direction_vector[2]

    p1 = (a * (v ** 2 + w ** 2) - u * (b * v + c * w - u * x - v * y - w * z)) * (
        1 - np.cos(angle_in_radians)) + x * np.cos(
            angle_in_radians) + (-c * v + b * w - w * y + v * z) * np.sin(angle_in_radians)

    p2 = (b * (u ** 2 + w ** 2) - v * (a * u + c * w - u * x - v * y - w * z)) * (
        1 - np.cos(angle_in_radians)) + y * np.cos(
            angle_in_radians) + (-c * u - a * w + w * x - u * z) * np.sin(angle_in_radians)

    p3 = (c * (u ** 2 + v ** 2) - w * (a * u + b * v - u * x - v * y - w * z)) * (
        1 - np.cos(angle_in_radians)) + z * np.cos(
            angle_in_radians) + (-b * u + a * v - v * x + u * y) * np.sin(angle_in_radians)

    return [p1, p2, p3]
