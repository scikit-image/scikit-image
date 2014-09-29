__all__ = ['polygon_clip', 'polygon_area']

import numpy as np


def polygon_clip(yp, xp, r0, c0, r1, c1):
    """Clip a polygon to the given bounding box.

    Parameters
    ----------
    yp, xp : (N,) ndarray of double
        Row and column coordinates of the polygon.
    (r0, c0), (r1, c1) : double
        Top-left and bottom-right coordinates of the bounding box.

    Returns
    -------
    y_clipped, x_clipped : (M,) ndarray of double
        Coordinates of clipped polygon.

    Notes
    -----
    This makes use of Sutherland-Hodgman clipping as implemented in
    AGG 2.4 and exposed in Matplotlib.

    """
    from matplotlib import path, transforms

    poly = path.Path(np.vstack((yp, xp)).T, closed=True)
    clip_rect = transforms.Bbox([[r0, c0], [r1, c1]])

    poly_clipped = poly.clip_to_bbox(clip_rect).to_polygons()[0]

    return poly_clipped[:, 0], poly_clipped[:, 1]


def polygon_area(py, px):
    """Compute the area of a polygon.

    Parameters
    ----------
    py, px : (N,) array of float
        Polygon coordinates.

    Returns
    -------
    a : float
        Area of the polygon.
    """
    py = np.asarray(py)
    px = np.asarray(px)
    return 0.5 * np.abs(np.sum((px[:-1] * py[1:]) - (px[1:] * py[:-1])))
