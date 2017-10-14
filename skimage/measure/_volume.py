from itertools import product
import numpy as np
from scipy import spatial


def _offsets_diamond(ndim):
    offsets = np.zeros((2 * ndim, ndim))
    for vertex, (axis, offset) in \
                        enumerate(product(range(ndim), (-0.5, 0.5))):
        offsets[vertex, axis] = offset
    return offsets


def _offsets_cube(ndim):
    offsets = np.zeros((2 ** ndim, ndim))
    for vertex, offset in enumerate(product(*[[-0.5, 0.5]] * ndim)):
        offsets[vertex] = offset
    return offsets


def expanded_convex_hull(points, reduce_inner_points=True,
                         square_pixels=False):
    """Return the ConvexHull of input points jiggled by 0.5 along all axes.

    If we think of pixels as having zero area, then a normal convex hull on
    pixel coordinates is sufficient to encapsulate input pixels. However, if
    we consider points as having an extent of 1 area or volume unit (as is
    done in ``regionprops.area``, for example), then such a convex hull
    underestimates the area/volume of the convex object. This function
    corrects for that by adding/subtracting 0.5 to each input point and
    adding that to the input set.

    Note: the function "expands" the initial point set either along a
    *single* axis at a time, resulting in a "diamond" shape drawn around
    each point, or *every* axis at a time (with ``square_pixels=True``),
    resulting in a "cube" shape around each point. The former results in
    an underestimate of the convex hull volume, while the latter results
    in an overestimate.

    Parameters
    ----------
    points : array, shape (npoints, ndim)
        Input coordinates.
    reduce_inner_points : bool, optional
        If True, an initial convex hull is computed, and only the simplices
        of this hull are jiggled and use to compute the larger hull.
    square_pixels : bool, optional
        If True, a pixel's extent is considered to include the hypercube
        of sidelength 1 centered on the pixel's coordinates. If False, it
        is instead counted as a diamond of diameter 1.

    Returns
    -------
    hull : `scipy.spatial.ConvexHull` object
        The convex hull of the expanded input points.
    """
    if reduce_inner_points:
        hull0 = spatial.ConvexHull(points)
        points = hull0.points[hull0.vertices]
    ndim = points.shape[1]
    # sums = list(itertools.product(*[(-0.5, 0, 0.5)] * ndim))
    if square_pixels:
        offsets = _offsets_cube(ndim)
    else:
        offsets = _offsets_diamond(ndim)
    expanded = (points[:, np.newaxis, :] + offsets).reshape(-1, ndim)
    hull = spatial.ConvexHull(expanded)
    return hull
