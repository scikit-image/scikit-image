import numpy as np
from scipy import ndimage as ndi
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


    if multichannel:
        # Convert image to float, otherwise if image is int, the order value in ndi.map_coordinates
        # will not be evaluated properly
        img = img_as_float(img)
        perp_lines = _line_profile_coordinates(src, dst, linewidth=linewidth)

        if img.ndim == 3:
            pixels = [ndi.map_coordinates(img[..., i], perp_lines, order=order, mode=mode, cval=cval) for i in range(img.shape[2])]
            pixels = np.transpose(np.asarray(pixels), (1, 2, 0))
        else:
            pixels = ndi.map_coordinates(img, perp_lines, order=order, mode=mode, cval=cval)

        intensities = pixels.mean(axis=1)
    else:
        # Convert 3d array to float, otherwise if array is int, the order value in ndi.map_coordinates
        # will not be evaluated properly
        img = img.astype(np.float)
        perp_lines = _line_profile_coordinates3d(src, dst, linewidth=linewidth)
        pixels = ndi.map_coordinates(img, perp_lines, order=order, mode=mode, cval=cval)
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
        since the line is 3d, a line with with would end up looking like a 3d cylinder.

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
    src_row, src_col, src_dim3 = src = np.asarray(src, dtype=float)
    dst_row, dst_col, dst_dim3 = dst = np.asarray(dst, dtype=float)
    d_row, d_col, d_dim3 = dst - src

    # Get one unit vector perpendicular to direction vector to find a point # Get 3d variance
    # that is one unit distance away from the destination vector (ex: ix+jy+kz=0, and we start with x = y = 1)
    # we subtract 1 from linewidth to change from pixel-counting
    # (make this line 3 pixels wide) to point distances (the
    # distance between pixel centers)
    if d_dim3 != 0:
        # try finding the solution to ix+jy+kz=0 for x = 1 and y = 1
        dimZ = - (d_row + d_col) / d_dim3  # 1i+1j+zk=0, find z
        lengthVector = np.sqrt(2 + dimZ**2)  # sqrt(1**2 + 1**2 + z**2)
        # Make into a unit vector to find unit vector displacement
        col_width = (linewidth - 1) * (1 / lengthVector) / 2
        row_width = (linewidth - 1) * (1 / lengthVector) / 2
        slice_width = (linewidth - 1) * (dimZ / lengthVector) / 2
    elif d_row != 0:
        # try finding the solution to ix+jy+kz=0 for y = 1 and z = 1
        dimX = - (d_dim3 + d_col) / d_row
        lengthVector = np.sqrt(2 + dimX**2)
        # Make into a unit vector to find unit vector displacement
        col_width = (linewidth - 1) * (1 / lengthVector) / 2
        row_width = (linewidth - 1) * (dimX / lengthVector) / 2
        slice_width = (linewidth - 1) * (1 / lengthVector) / 2
    else:
        # try finding the solution to ix+jy+kz=0 for x = 1 and z = 1
        dimY = - (d_row + d_dim3) / d_col
        lengthVector = np.sqrt(2 + dimY**2)
        # Make into a unit vector to find unit vector displacement
        col_width = (linewidth - 1) * (dimY / lengthVector) / 2
        row_width = (linewidth - 1) * (1 / lengthVector) / 2
        slice_width = (linewidth - 1) * (1 / lengthVector) / 2

    # we add one above because we include the last point in the profile
    # (in contrast to standard numpy indexing)
    length = np.ceil(np.linalg.norm([d_row, d_col, d_dim3]) + 1)
    unitDirectionVector = [d_row / (length - 1), d_col / (length - 1), d_dim3 / (length - 1)]

    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)
    line_dim3 = np.linspace(src_dim3, dst_dim3, length)

    perp_rows = np.array([np.linspace(row_i - row_width, row_i + row_width, linewidth) for row_i in line_row])
    perp_cols = np.array([np.linspace(col_i - col_width, col_i + col_width, linewidth) for col_i in line_col])
    perp_slices = np.array([np.linspace(slice_i - slice_width, slice_i + slice_width, linewidth) for slice_i in line_dim3])

    perp_Array = np.array([perp_rows, perp_cols, perp_slices])

    if linewidth == 1:
        return perp_Array

    # Rotate the points around the vector, the number of times depends on the linewidth
    # for example, for a linewidth of 2 or 3, only one rotation of 90 degrees for is necessary to the extra points, more
    # rotations would not bring extra accuracy
    # For linewidth = 4: 3 rotations of 45 degrees ( +45, +90 and + 135) and the resulting points are added)
    # For linewidth = 5: 6 rotations of 45/2 degrees
    # the center point will not be reproduced, as the rotation around the vector doesn't do anything for that point.

    # find array of rotations needed depending on line width
    angleArray = np.linspace(0, 180, (linewidth * 2) - 1) #
    # Remove first and last elements, which are 0 and 180, because there are already points there
    angleArray = np.delete(angleArray, len(angleArray)-1)
    #angleArray = np.delete(angleArray, 0)

    transposedArray = perp_Array.T
    returnArray = []

    # loop through all the angles to get the transformation
    for j in range(transposedArray.shape[0]): # the number of sample points per unit (i.e. linewidth)
        for angle in angleArray:
            pointArray = []
            for i in range(transposedArray.shape[1]): # the number of unit points on displacement vector
                point = transposedArray[j][i]
                if angle == 0:
                    pointArray.append(point)
                else:
                    # Check to see if point is on axis (i.e. when linewidth is odd)
                    # and ignore, since it has already been added for rotation = 0
                    if distancePointLine(point, src, dst) == 0:
                        continue
                    rotatedPoint = np.array(Rotate(point, src, unitDirectionVector, np.radians(angle)))
                    pointArray.append(rotatedPoint)

            if len(pointArray) > 0: # check because it's empty arrays will come through
                returnArray.append(pointArray)

    returnArray = np.array(returnArray, dtype=float)

    # Remove duplicate elements, which can happen if the linewidth was odd
    #returnArray = np.unique(returnArray)

    # return result transposed for mapping - must be of shape [3, 3, x] where x is the linewidth and
    # x is the number of sample points per unit point along vector
    return returnArray.T

# Determines if a point is on a line
def distancePointLine(point, src, dst):
    lineVector = np.subtract(dst, src)
    #lengthLine = np.linalg.norm([dst[0]-src[0], dst[1]-src[1], dst[2]-src[2]])
    lengthLine = np.linalg.norm(lineVector)
    vector1 = np.subtract(point, dst)
    vector2 = np.subtract(point, src)
    distance = np.linalg.norm(np.cross(vector1, vector2) / lengthLine)
    return np.abs(distance)

# Code to rotate point around axis
def Rotate(pointToRotate, pointOfLine, directionVector, theta):
    a = pointOfLine[0]
    b = pointOfLine[1]
    c = pointOfLine[2]

    x = pointToRotate[0]
    y = pointToRotate[1]
    z = pointToRotate[2]

    u = directionVector[0]
    v = directionVector[1]
    w = directionVector[2]

    p1 = (a * (v ** 2 + w ** 2) - u * (b * v + c * w - u * x - v * y - w * z)) * (1 - np.cos(theta)) + x * np.cos(
            theta) + (-c * v + b * w - w * y + v * z) * np.sin(theta)

    p2 = (b * (u ** 2 + w ** 2) - v * (a * u + c * w - u * x - v * y - w * z)) * (1 - np.cos(theta)) + y * np.cos(
            theta) + (-c * u - a * w + w * x - u * z) * np.sin(theta)

    p3 = (c * (u ** 2 + v ** 2) - w * (a * u + b * v - u * x - v * y - w * z)) * (1 - np.cos(theta)) + z * np.cos(
            theta) + (-b * u + a * v - v * x + u * y) * np.sin(theta)

    return [p1, p2, p3]
