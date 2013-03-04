__all__ = ['convex_hull_image']

import numpy as np
from ._pnpoly import grid_points_inside_poly
from ._convex_hull import possible_hull


def convex_hull_image(image):
    """Compute the convex hull image of a binary image.

    The convex hull is the set of pixels included in the smallest convex
    polygon that surround all white pixels in the input image.

    Parameters
    ----------
    image : ndarray
        Binary input image.  This array is cast to bool before processing.

    Returns
    -------
    hull : ndarray of uint8
        Binary image with pixels in convex hull set to 255.

    References
    ----------
    .. [1] http://blogs.mathworks.com/steve/2011/10/04/binary-image-convex-hull-algorithm-notes/

    """

    image = image.astype(bool)

    # Here we do an optimisation by choosing only pixels that are
    # the starting or ending pixel of a row or column.  This vastly
    # limits the number of coordinates to examine for the virtual
    # hull.
    coords = possible_hull(image.astype(np.uint8))
    N = len(coords)

    # Add a vertex for the middle of each pixel edge
    coords_corners = np.empty((N * 4, 2))
    for i, (x_offset, y_offset) in enumerate(zip((0, 0, -0.5, 0.5),
                                                 (-0.5, 0.5, 0, 0))):
        coords_corners[i * N:(i + 1) * N] = coords + [x_offset, y_offset]

    coords = coords_corners

    try:
        from scipy.spatial import Delaunay
    except ImportError:
        raise ImportError('Could not import scipy.spatial, only available in '
                          'scipy >= 0.9.')

    # Find the convex hull
    chull = Delaunay(coords).convex_hull
    v = coords[np.unique(chull)]

    # Sort vertices clock-wise
    v_centred = v - v.mean(axis=0)
    angles = np.arctan2(v_centred[:, 0], v_centred[:, 1])
    v = v[np.argsort(angles)]

    # For each pixel coordinate, check whether that pixel
    # lies inside the convex hull
    mask = grid_points_inside_poly(image.shape[:2], v)

    return mask
