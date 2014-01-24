import numpy as np
import scipy.ndimage as ndi

def _calc_vert(img, x1, x2, y1, y2, linewidth):
    # Quick calculation if perfectly horizontal
    pixels = img[min(y1, y2): max(y1, y2) + 1,
                 x1 - linewidth / 2: x1 + linewidth / 2 + 1]

    # Reverse index if necessary
    if y2 > y1:
        pixels = pixels[::-1, :]

    return pixels.mean(axis=1)[:, np.newaxis]


def profile_line(img, end_points, linewidth=1, mode='constant', cval=0.0):
    """Return the intensity profile of an image measured along a scan line.

    Parameters
    ----------
    img : 2d or 3d array
        The image, in grayscale (2d) or RGB (3d) format.
    end_points : (2, 2) list
        End points ((x1, y1), (x2, y2)) of scan line.
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
    >>> profile_line(img, ((1, 2), (5, 2)))
    array([[ 1.],
           [ 1.],
           [ 2.],
           [ 2.]])
    """
    point1, point2 = end_points
    x1, y1 = point1 = np.asarray(point1, dtype=float)
    x2, y2 = point2 = np.asarray(point2, dtype=float)
    dx, dy = point2 - point1
    channels = 1
    if img.ndim == 3:
        channels = 3

    # Quick calculation if perfectly vertical; shortcuts div0 error
    if x1 == x2:
        if channels == 1:
            img = img[:, :, np.newaxis]

        img = np.rollaxis(img, -1)
        intensities = np.hstack([_calc_vert(im, x1, x2, y1, y2, linewidth)
                                 for im in img])
        return intensities

    theta = np.arctan2(dy, dx)
    a = dy / dx
    b = y1 - a * x1
    length = np.hypot(dx, dy)

    line_x = np.linspace(x1, x2, np.ceil(length))
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

    if intensities.ndim == 1:
        return intensities[..., np.newaxis]
    else:
        return intensities

