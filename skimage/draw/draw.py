# coding: utf-8

from __future__ import division
from itertools import izip

import numpy as np


def _coords_inside_image(rr, cc, shape):
    mask = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    return rr[mask], cc[mask]


def _pairs(iterable):
    "s -> (s0,s1), (s2,s3), (s4, s5), ..."
    a = iter(iterable)
    return izip(a, a)


def polygon_scanline(image, yp, xp):
    """Draw polygon onto image using a scanline algorithm.

    Attributes
    ----------
    yp, xp : double array
        Coordinates of polygon.

    References
    ----------
    .. [1] "Intersection point of two lines in 2 dimensions",
        http://paulbourke.net/geometry/pointlineplane/
    .. [2] UC Davis ECS175 (Introduction to Computer Graphics) notes,
       http://www.cs.ucdavis.edu/~ma/ECS175_S00/Notes/0411_b.pdf
    """
    yp = list(yp)
    xp = list(xp)

    y_start, y_end = np.min(yp), np.max(yp)
    if not ((yp[0] == yp[-1]) and (xp[0] == xp[-1])):
        yp.append(yp[0])
        xp.append(xp[0])

    ys = zip(yp[:-1], yp[1:])
    xs = zip(xp[:-1], xp[1:])

    h, w = image.shape[:2]

    segments = zip(xs, ys)

    for y in range(max(0, y_start), min(h, y_end)):
        intersections = []

        for ((x0, x1), (y0, y1)) in segments:
            if y0 == y1:
                continue

            xmin = min(x0, x1)
            xmax = max(x0, x1)
            ymin = min(y0, y1)
            ymax = max(y0, y1)

            if not (ymin <= y <= ymax):
                    continue

            xi = ((x1 - x0) * (y - y0) - (y1 - y0) * (-x0)) / (y1 - y0)

            if (xmin <= xi <= xmax):
                if (y == y0 or y == y1) and (y != ymin):
                    continue
                intersections.append(xi)

        intersections = np.sort(intersections)

        for x0, x1 in _pairs(intersections):
            image[y, max(0, np.ceil(x0)):min(np.ceil(x1), w)] = 1

    return image


def ellipse(cy, cx, yradius, xradius, shape=None):
    """Generate coordinates of pixels within ellipse.

    Parameters
    ----------
    cy, cx : double
        Centre coordinate of ellipse.
    yradius, xradius : double
        Minor and major semi-axes. ``(x/xradius)**2 + (y/yradius)**2 = 1``.
    shape : tuple, optional
        Image shape which is used to determine maximum extents of output pixel
        coordinates. This is useful for ellipses which exceed the image size.
        By default the full extents of the ellipse are used.

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of ellipse.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Examples
    --------
    >>> from skimage.draw import ellipse
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = ellipse(5, 5, 3, 4)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    """

    dr = 1 / float(yradius)
    dc = 1 / float(xradius)

    r, c = np.ogrid[-1:1:dr, -1:1:dc]
    rr, cc = np.nonzero(r ** 2 + c ** 2 < 1)

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
        Image shape which is used to determine maximum extents of output pixel
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

    Examples
    --------
    >>> from skimage.draw import circle
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = circle(4, 4, 5)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

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

    Examples
    --------
    >>> from skimage.draw import line, set_color
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = line(1, 1, 20, 20)
    >>> set_color(img, (rr, cc), 1)
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=uint8)

    """

    rr, cc = coords
    rr, cc = _coords_inside_image(rr, cc, img.shape)
    img[rr, cc] = color


if __name__ == "__main__":
    image = np.zeros((100, 100))

    p = np.array([[20, 5],
                  [0, 20],
                  [10, 16],
                  [20, 20],
                  [40, 40],
                  [19, 15],
                  [20, 5]])

    import matplotlib.pyplot as plt

    polygon_scanline(image, p[:, 1], p[:, 0])

    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.plot(p[:, 0], p[:, 1], 'r-o')
    plt.axis('image')
    plt.show()
