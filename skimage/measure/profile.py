import numpy as np
import scipy.ndimage as ndi

def _calc_vert(img, col, src_row, dst_row, linewidth):
    # Quick calculation if perfectly vertical
    pixels = img[src_row:dst_row:np.sign(dst_row - src_row),
                 col - linewidth / 2: col + linewidth / 2 + 1]
    return pixels.mean(axis=1)[..., np.newaxis]


def profile_line(img, src, dst, linewidth=1, mode='constant', cval=0.0):
    """Return the intensity profile of an image measured along a scan line.

    Parameters
    ----------
    img : 2d or 3d array
        The image, in grayscale (2d) or RGB (3d) format.
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
    array([ 1., 1., 2., 2.])
    """
    src_row, src_col = src = np.asarray(src, dtype=float)
    dst_row, dst_col = dst = np.asarray(dst, dtype=float)
    d_row, d_col = dst - src

    # Quick calculation if perfectly vertical; shortcuts div0 error
    if src_col == dst_col:
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        img = np.rollaxis(img, -1)
        intensities = np.hstack([_calc_vert(im, src_col, 
                                            src_row, dst_row, linewidth)
                                 for im in img])
        return np.squeeze(intensities)

    theta = np.arctan2(d_row, d_col)
    a = d_row / d_col
    b = src_row - a * src_col
    length = np.hypot(d_row, d_col)

    line_x = np.linspace(src_col, dst_col, np.ceil(length))
    line_y = line_x * a + b
    y_width = abs(linewidth * np.cos(theta) / 2)
    perp_ys = np.array([np.linspace(yi - y_width,
                                    yi + y_width, linewidth) for yi in line_y])
    perp_xs = - a * perp_ys + (line_x + a * line_y)[:, np.newaxis]

    perp_lines = np.array([perp_ys, perp_xs])
    if img.ndim == 3:
        pixels = [ndi.map_coordinates(img[..., i], perp_lines,
                                      mode=mode, cval=cval) for i in range(3)]
        pixels = np.transpose(np.asarray(pixels), (1, 2, 0))
    else:
        pixels = ndi.map_coordinates(img, perp_lines, mode=mode, cval=cval)
        pixels = pixels[..., np.newaxis]

    intensities = pixels.mean(axis=1)

    return intensities

