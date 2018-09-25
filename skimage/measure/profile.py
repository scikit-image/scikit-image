import math
import numpy as np
from scipy import ndimage as ndi
from ._linalg import rotate_sample_points


def profile_line(image, src, dst, linewidth=1,
                 order=1, mode='constant', cval=0.0, multichannel=True):
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

    perp_lines = _line_profile_coordinates(src, dst, linewidth=linewidth)
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


def _line_profile_coordinates(src, dst, linewidth=1):
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
    if len(src) == 2:
        src_row, src_col = src = np.asarray(src, dtype=float)
        dst_row, dst_col = dst = np.asarray(dst, dtype=float)
        d_row, d_col = dst - src
        theta = np.arctan2(d_row, d_col)

        length = math.ceil(np.hypot(d_row, d_col) + 1)
        # we add one above because we include the last point in the profile
        # (in contrast to standard numpy indexing)
        line_col = np.linspace(src_col, dst_col, length)
        line_row = np.linspace(src_row, dst_row, length)

        # we subtract 1 from linewidth to change from pixel-counting
        # (make this line 3 pixels wide) to point distances (the
        # distance between pixel centers)
        col_width = (linewidth - 1) * np.sin(-theta) / 2
        row_width = (linewidth - 1) * np.cos(theta) / 2
        perp_rows = [np.linspace(row_i - row_width, row_i + row_width, linewidth) for row_i in line_row]
        perp_cols = [np.linspace(col_i - col_width, col_i + col_width, linewidth) for col_i in line_col]
        return np.array([perp_rows, perp_cols])

    elif len(src) == 3:
        src_pln, src_row, src_col = src = np.asarray(src, dtype=float)
        dst_pln, dst_row, dst_col = dst = np.asarray(dst, dtype=float)
        d_pln, d_row, d_col = dst - src

        # Get one unit vector perpendicular to direction vector to find a point
        # that is one unit distance away from the destination vector
        # (ex: ix + jy + kz = 0, then we can use x = y = 1)
        # Try with z first if it is not 0, then the same for x, otherwise pick y
        # We subtract 1 from linewidth to change from pixel-counting
        # (make this line 3 pixels wide) to point distances (the
        # distance between pixel centers)
        if d_pln != 0:
            # try finding the solution to ix + jy + kz = 0 for x = 1 and y = 1
            dim_z = - (d_row + d_col) / d_pln
            length_vector = np.sqrt(2 + dim_z ** 2)
            col_width = row_width = (linewidth - 1) / (2 * length_vector)
            slice_width = (linewidth - 1) * (dim_z / 2 * length_vector)
        elif d_row != 0:
            # try finding the solution to ix + jy + kz = 0 for y = 1 and z = 1
            dim_x = - (d_pln + d_col) / d_row
            length_vector = np.sqrt(2 + dim_x ** 2)
            col_width = slice_width = (linewidth - 1) / (2 * length_vector)
            row_width = (linewidth - 1) * (dim_x / length_vector) / 2
        else:
            # try finding the solution to ix + jy + kz = 0 for x = 1 and z = 1
            dim_y = - (d_row + d_pln) / d_col
            length_vector = np.sqrt(2 + dim_y ** 2)
            row_width = slice_width = (linewidth - 1) / (2 * length_vector)
            col_width = (linewidth - 1) * (dim_y / length_vector) / 2

        # we add one above because we include the last point in the profile
        # (in contrast to standard numpy indexing)
        length = math.ceil(np.linalg.norm([d_pln, d_row, d_col]) + 1)

        line_col = np.linspace(src_col, dst_col, length)
        line_row = np.linspace(src_row, dst_row, length)
        line_pln = np.linspace(src_pln, dst_pln, length)

        # find divisor to get only first half of array and center point if odd
        # first_half_index = np.int(np.ceil(linewidth/2))
        # if first_half_index < 1:
        #     first_half_index = 1
        #
        # perp_rows = [np.linspace(row_i - row_width, row_i + row_width, linewidth)[:first_half_index] for row_i in line_row]
        # perp_cols = [np.linspace(col_i - col_width, col_i + col_width, linewidth)[:first_half_index] for col_i in line_col]
        # perp_pln = [np.linspace(pln_i - slice_width, pln_i + slice_width, linewidth)[:first_half_index] for pln_i in line_pln]

        perp_rows = [np.linspace(row_i - row_width, row_i + row_width, linewidth) for row_i in line_row]
        perp_cols = [np.linspace(col_i - col_width, col_i + col_width, linewidth) for col_i in line_col]
        perp_pln = [np.linspace(pln_i - slice_width, pln_i + slice_width, linewidth) for pln_i in line_pln]

        perp_array = np.array([perp_pln, perp_rows, perp_cols])

        if linewidth > 1:
            # create a rotated array of sample points around the direction axis to make the line width 3D
            perp_array = rotate_sample_points(linewidth, perp_array, src, dst)

        return perp_array
