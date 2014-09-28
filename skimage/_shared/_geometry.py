__all__ = ['polygon_clip', 'polygon_area']

import numpy as np


def polygon_clip(rr, cc, r0, c0, r1, c1):
    """Clip a polygon to the given bounding box.

    Parameters
    ----------
    yp, xp : (N,) ndarray of double
        Row and column coordinates of the polygon.
    ytop, xleft, ybottom, xright : double
        Coordinates of the bounding box.  Note that the following
        must hold true: ``x_left < x_right`` and ``y_top < y_bottom``.

    Returns
    -------
    y_clipped, x_clipped : (M,) ndarray of double
        Coordinates of clipped polygon.

    Notes
    -----
    The algorithm is a translation of the Pascal code found in [1]_ and
    includes fixes from Anti-Grain Geometry v2.4.

    References
    ----------
    .. [1] You-Dong Liang and Brian A. Barsky,
           An Analysis and Algorithm for Polygon Clipping,
           Communications of the ACM, Vol 26, No 11, November 1983.

    """
    from matplotlib import path, transforms

    poly = path.Path(np.vstack((rr, cc)).T, closed=True)
    clip_rect = transforms.Bbox([[r0, c0], [r1, c1]])

    poly_clipped = poly.clip_to_bbox(clip_rect).to_polygons()[0]

    return poly_clipped[:, 0], poly_clipped[:, 1]


def polygon_area(py, px):
    """Compute the area of a polygon.

    Parameters
    ----------
    py, px : (N,) array of float
        Polygon coordinates.
    """
    py = np.asarray(py)
    px = np.asarray(px)
    return 0.5 * np.abs(np.sum((px[:-1] * py[1:]) - (px[1:] * py[:-1])))
