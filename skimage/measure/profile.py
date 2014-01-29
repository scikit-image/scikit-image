import numpy as np
import scipy.ndimage as ndi


def profile_line(img, src, dst, linewidth=1, mode='constant', cval=0.0):
    """Return the intensity profile of an image measured along a scan line.

    Parameters
    ----------
    img : 2d or 3d array
        The image, in grayscale (2d) or multichannel (2d + c) format.
    src : 2-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 2-tuple of numeric scalar (float or int)
        The end point of the scan line.
    linewidth : int, optional
        Width of the scan, perpendicular to the line
    mode : string, one of {'constant', 'nearest', 'reflect', 'wrap'}, optional
        How to compute any values falling outside of the image.
    cval : float, optional
        If `mode` is 'constant', what constant value to use outside the image.

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
    >>> profile_line(img, (2, 1), (2, 5))
    array([ 1.,  1.,  2.,  2.])
    """
    src_row, src_col = src = np.asarray(src, dtype=float)
    dst_row, dst_col = dst = np.asarray(dst, dtype=float)
    d_row, d_col = dst - src

    if d_col == 0:
        if d_row > 0:
            theta = -np.pi / 2
        else:
            theta = np.pi / 2
    else:
        theta = np.arctan2(-d_row, d_col)

    length = np.ceil(np.hypot(d_row, d_col))
    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)

    # this if clause is necessary to keep the line centered on the true
    # source and destination points. Otherwise, the computed line has
    # an offset of `linewidth/2`
    if linewidth <= 1:
        perp_lines = np.array([line_row[:, np.newaxis],
                               line_col[:, np.newaxis]])
    else:
        col_width = linewidth * np.sin(theta) / 2
        row_width = linewidth * np.cos(theta) / 2
        perp_rows = np.array([np.linspace(row_i - row_width, row_i + row_width,
                                          linewidth) for row_i in line_row])
        perp_cols = np.array([np.linspace(col_i - col_width, col_i + col_width,
                                          linewidth) for col_i in line_col])
        perp_lines = np.array([perp_rows, perp_cols])
    if img.ndim == 3:
        pixels = [ndi.map_coordinates(img[..., i], perp_lines, mode=mode,
                                      cval=cval) for i in range(img.shape[2])]
        pixels = np.transpose(np.asarray(pixels), (1, 2, 0))
    else:
        pixels = ndi.map_coordinates(img, perp_lines, mode=mode, cval=cval)

    intensities = pixels.mean(axis=1)

    return intensities

