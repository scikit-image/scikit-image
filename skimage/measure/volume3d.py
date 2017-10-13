import itertools
import numpy as np
from scipy import spatial


def extended_convex_hull(points, reduce_inner_points=True):
    """Return the ConvexHull of input points jiggled by 0.5 along all axes.

    If we think of pixels as having zero area, then a normal convex hull
    on pixel coordinates is sufficient to encapsulate input pixels. However,
    if we consider points as having an extent of 1 area or volume unit (as is
    done in ``regionprops.area``, for example), then such a convex hull
    underestimates the area/volume of the convex object. This function corrects
    for that by adding/subtracting 0.5 to each input point and adding that to
    the input set.

    Parameters
    ----------
    points : array, shape (npoints, ndim)
        Input coordinates.
    reduce_inner_points : bool, optional
        If True, an initial convex hull is computed, and only the simplices
        of this hull are jiggled and use to compute the larger hull.

    Returns
    -------
    hull : `scipy.spatial.ConvexHull` object
    """
    if reduce_inner_points:
        hull0 = spatial.ConvexHull(points)
        points = hull0.points[hull0.vertices]
    ndim = points.shape[1]
    sums = list(itertools.combinations_with_replacement((-0.5, 0, 0.5), ndim))
    jittered = (points[:, np.newaxis, :] + sums).reshape(-1, ndim)
    hull = spatial.ConvexHull(jittered)
    return hull
