import math
import numpy as np
from scipy import ndimage as ndi, constants

from ._linalg import distance_point_line, rotate_point_around_line, get_any_perpendicular_vector


def profile_line(image, src, dst, linewidth=1,
                 order=1, mode='constant', cval=0.0, multichannel=True,
                 endpoint=True):
    """Return the intensity profile of an image measured along a scan line.

    Parameters
    ----------
    image : numeric array, shape (M, N[, C])
        The image, either grayscale (2D or 3D array) or multichannel
        (3D array, for a RBG 2D image where the final axis contains the channel
        information).
    src : 2-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 2-tuple of numeric scalar (float or int)
        The end point of the scan line. The destination point is *included*
        in the profile, in contrast to standard numpy indexing.
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
        Whether the last axis of the image is to be interpreted as RGB
        channels or another spatial dimension.
    endpoint : bool, optional
        If True, returns the intensity value at dst. Otherwise, it is not included.
        Default is True.

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
    >>> profile_line(img, (1, 0), (1, 6), cval=4)
    array([ 1.,  1.,  1.,  2.,  2.,  2.,  4.])

    The destination point is included in the profile, in contrast to
    standard numpy indexing.
    For example:

    >>> profile_line(img, (1, 0), (1, 6))  # The final point is out of bounds
    array([ 1.,  1.,  1.,  2.,  2.,  2.,  0.])
    >>> profile_line(img, (1, 0), (1, 5))  # This accesses the full first row
    array([ 1.,  1.,  1.,  2.,  2.,  2.])
    """
    if image.ndim not in [2, 3, 4]:
        raise ValueError('profile_line is not implemented for images of dimension {0}'.format(image.shape))

    perp_lines = _line_profile_coordinates(src, dst, linewidth=linewidth, endpoint=endpoint)
    if image.ndim == 4 or (image.ndim == 3 and multichannel):
        # 2D or 3D multichannel
        pixels = [ndi.map_coordinates(image[..., i], perp_lines, order=order,
                                      mode=mode, cval=cval) for i in range(image.shape[image.ndim - 1])]
        pixels = np.transpose(np.asarray(pixels), (1, 2, 0))
    else:
        # 2D or 3D grayscale
        pixels = ndi.map_coordinates(image, perp_lines, order=order, mode=mode, cval=cval)

    intensities = pixels.mean(axis=1)
    return intensities


def _line_profile_coordinates(src, dst, linewidth=1, endpoint=True):
    """Return the coordinates of the profile of an image along a scan line.

    Parameters
    ----------
    src : 2 or 3-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 2 or 3-tuple of numeric scalar (float or int)
        The end point of the scan line.
    linewidth : int, optional
        Width of the scan, perpendicular to the line. In 3D, this value is the
        diameter of a 3d cylinder along the scan line.

    Returns
    -------
    perp_array : array, shape (2 or 3, N, C), float
        The coordinates of the profile along the scan line. The length of the
        profile is the ceil of the computed length of the scan line.

    Notes
    -----
    This is a utility method meant to be used internally by skimage functions.
    The destination point is included in the profile, in contrast to
    standard numpy indexing.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    length = math.ceil(np.linalg.norm(dst - src))

    # when endpoint is true add 1 to length to include the last point in the profile.
    # (in contrast to standard numpy indexing)
    num = length + 1 if endpoint else length

    if len(src) == 2:
        line_row = np.linspace(src[0], dst[0], num, endpoint=endpoint)
        line_col = np.linspace(src[1], dst[1], num, endpoint=endpoint)
        d_row, d_col = (dst - src) / length
        # we subtract 1 from linewidth to change from pixel-counting
        # (make this line 3 pixels wide) to point distances (the
        # distance between pixel centers)
        row_width = (linewidth - 1) * d_col / 2
        col_width = (linewidth - 1) * -d_row / 2
        perp_rows = [np.linspace(row_i - row_width, row_i + row_width, linewidth) for row_i in line_row]
        perp_cols = [np.linspace(col_i - col_width, col_i + col_width, linewidth) for col_i in line_col]
        return np.array([perp_rows, perp_cols])

    elif len(src) == 3:
        line_pln = np.linspace(src[0], dst[0], num, endpoint=endpoint)
        line_row = np.linspace(src[1], dst[1], num, endpoint=endpoint)
        line_col = np.linspace(src[2], dst[2], num, endpoint=endpoint)
        d_pln, d_row, d_col = (dst - src) / length
        perp_vector = np.asarray(get_any_perpendicular_vector([d_pln, d_row, d_col]))
        pln_width, row_width, col_width,  = (linewidth - 1) * perp_vector / 2
        #row_width, col_width, pln_width = perp_vector
        #return v2[0], v2[1], v2[2]

        #col_width, row_width, pln_width = get_abc(d_pln, d_row, d_col, linewidth)
        # row_width = (linewidth - 1) * (d_col / length) / 2
        # col_width = (linewidth - 1) * -(d_row / length) / 2
        # pln_width = (linewidth - 1) * (d_pln / length) / 2

        # find divisor to get only first half of array and center point if odd
        # first_half_index = np.int(np.ceil(linewidth/2))
        # if first_half_index < 1:
        #     first_half_index = 1
        #
        # perp_rows = [np.linspace(row_i - row_width, row_i + row_width, linewidth)[:first_half_index] for row_i in line_row]
        # perp_cols = [np.linspace(col_i - col_width, col_i + col_width, linewidth)[:first_half_index] for col_i in line_col]
        # perp_pln = [np.linspace(pln_i - slice_width, pln_i + slice_width, linewidth)[:first_half_index] for pln_i in line_pln]


        if linewidth > 1:
            # create a rotated array of sample points around the direction axis to make the line width 3D
            perp_array = _rotate_sample_points(linewidth, perp_array, src, dst)


        perp_pln = [np.linspace(pln_i - pln_width, pln_i + pln_width, linewidth) for pln_i in line_pln]
        perp_rows = [np.linspace(row_i - row_width, row_i + row_width, linewidth) for row_i in line_row]
        perp_cols = [np.linspace(col_i - col_width, col_i + col_width, linewidth) for col_i in line_col]

        perp_array = np.array([perp_pln, perp_rows, perp_cols])

        # if linewidth > 1:
        #     # create a rotated array of sample points around the direction axis to make the line width 3D
        #     perp_array = _rotate_sample_points(linewidth, perp_array, src, dst)

        return perp_array


def _rotate_sample_points(perp_array, src, dst, number_of_sample_points=180):
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
    unit_direction_vector = np.divide(np.subtract(dst, src), np.linalg.norm(line_vector))

    # Rotate the points around the axis a number of times depending on the distance of the point
    # from the direction axis to simulate sampling of points around the axis
    sampling_array = []
    for perp_points in np.transpose(perp_array):
        #rotation_angles = _rotation_angles_by_distance_from_line(dst, src, perp_points[0])
        #distance = distance_point_line(point, src, dst)
        # if distance == 0:
        #     # the point is on the line
        #     pass
        # Return an array of angles that will be used to rotate a point 360 degrees around a line
        rotation_angles = np.linspace(0, 2 * constants.pi, number_of_angles, endpoint=False)

        for angle in rotation_angles:  # the number of angles to use as rotation angles for the sampling points
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
