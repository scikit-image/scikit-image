# coding: utf-8
import numpy as np


def _coords_inside_image(rr, cc, shape):
    mask = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    return rr[mask], cc[mask]


def ellipse(cy, cx, yradius, xradius, shape=None):
    """Generate coordinates of pixels within ellipse.

    Parameters
    ----------
    cy, cx : double
        Centre coordinate of ellipse.
    yradius, xradius : double
        Minor and major semi-axes. ``(x/xradius)**2 + (y/yradius)**2 = 1``.
    shape : tuple, optional
        image shape which is used to determine maximum extents of output pixel
        coordinates. This is useful for ellipses which exceed the image size.
        By default the full extents of the ellipse are used.

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of ellipse.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    """

    dr = 1 / float(yradius)
    dc = 1 / float(xradius)

    r, c = np.ogrid[-1:1:dr, -1:1:dc]
    rr, cc = np.nonzero(r ** 2  + c ** 2 < 1)

    rr.flags.writeable = True
    cc.flags.writeable = True
    rr += cy - yradius
    cc += cx - xradius

    if shape is not None:
        _coords_inside_image(rr, cc, shape)

    return rr, cc


def circle(cy, cx, radius, shape=None):
    """Generate coordinates of pixels within circle.

    Parameters
    ----------
    cy, cx : double
        Centre coordinate of circle.
    radius: double
        Radius of circle.
    shape : tuple, optional
        image shape which is used to determine maximum extents of output pixel
        coordinates. This is useful for circles which exceed the image size.
        By default the full extents of the circle are used.

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of circle.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.
    Notes
    -----
        This function is a wrapper for skimage.draw.ellipse()
    """

    return ellipse(cy, cx, radius, radius, shape)


def set_color(img, coords, color):
    """Set pixel color in the image at the given coordinates.

    Coordinates that exceed the shape of the image will be ignored.

    Parameters
    ----------
    img : (M, N, D) ndarray
        Image
    coords : ((P,) ndarray, (P,) ndarray)
        Coordinates of pixels to be colored.
    color : (D,) ndarray
        Color to be assigned to coordinates in the image.

    Returns
    -------
    img : (M, N, D) ndarray
        The updated image.

    """

    rr, cc = coords
    rr, cc = _coords_inside_image(rr, cc, img.shape)
    img[rr, cc] = color
