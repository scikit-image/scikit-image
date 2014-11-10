__all__ = ['polygon_clip', 'polygon_area']

import numpy as np


def polygon_clip(rp, cp, r0, c0, r1, c1):
    """Clip a polygon to the given bounding box.

    Parameters
    ----------
    rp, cp : (N,) ndarray of double
        Row and column coordinates of the polygon.
    (r0, c0), (r1, c1) : double
        Top-left and bottom-right coordinates of the bounding box.

    Returns
    -------
    r_clipped, c_clipped : (M,) ndarray of double
        Coordinates of clipped polygon.

    Notes
    -----
    This makes use of Sutherland-Hodgman clipping as implemented in
    AGG 2.4 and exposed in Matplotlib.

    """
    from matplotlib import _path, path, transforms

    # `clip_to_bbox` is included directly from Matplotlib
    # since it was only included after v1.1
    def clip_to_bbox(poly_path, bbox, inside=True):
        verts = _path.clip_path_to_rect(poly_path, bbox, inside)
        paths = [path.Path(poly) for poly in verts]
        return poly_path.make_compound_path(*paths)

    poly = path.Path(np.vstack((rp, cp)).T, closed=True)
    clip_rect = transforms.Bbox([[r0, c0], [r1, c1]])

    poly_clipped = clip_to_bbox(poly, clip_rect).to_polygons()[0]

    return poly_clipped[:, 0], poly_clipped[:, 1]


def polygon_area(pr, pc):
    """Compute the area of a polygon.

    Parameters
    ----------
    pr, pc : (N,) array of float
        Polygon row and column coordinates.

    Returns
    -------
    a : float
        Area of the polygon.
    """
    pr = np.asarray(pr)
    pc = np.asarray(pc)
    return 0.5 * np.abs(np.sum((pc[:-1] * pr[1:]) - (pc[1:] * pr[:-1])))
