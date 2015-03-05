__all__ = ['convex_hull_image', 'convex_hull_object']

import numpy as np
from ..measure import grid_points_in_poly
from ._convex_hull import possible_hull
from ..measure._label import label
from ..util import unique_rows


def convex_hull_image(image):
    """Compute the convex hull image of a binary image.

    The convex hull is the set of pixels included in the smallest convex
    polygon that surround all white pixels in the input image.

    Parameters
    ----------
    image : ndarray
        Binary input image. This array is cast to bool before processing.

    Returns
    -------
    hull : ndarray of bool
        Binary image with pixels in convex hull set to True.

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

    # repeated coordinates can *sometimes* cause problems in
    # scipy.spatial.Delaunay, so we remove them.
    coords = unique_rows(coords_corners)

    try:
        from scipy.spatial import Delaunay
    except ImportError:
        raise ImportError('Could not import scipy.spatial, only available in '
                          'scipy >= 0.9.')

    # Subtract offset
    offset = coords.mean(axis=0)
    coords -= offset

    # Find the convex hull
    chull = Delaunay(coords).convex_hull
    v = coords[np.unique(chull)]

    # Sort vertices clock-wise
    v_centred = v - v.mean(axis=0)
    angles = np.arctan2(v_centred[:, 0], v_centred[:, 1])
    v = v[np.argsort(angles)]

    # Add back offset
    v += offset

    # For each pixel coordinate, check whether that pixel
    # lies inside the convex hull
    mask = grid_points_in_poly(image.shape[:2], v)

    return mask


def convex_hull_object(image, neighbors=8):
    """Compute the convex hull image of individual objects in a binary image.

    The convex hull is the set of pixels included in the smallest convex
    polygon that surround all white pixels in the input image.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    neighbors : {4, 8}, int
        Whether to use 4- or 8-connectivity.

    Returns
    -------
    hull : ndarray of bool
        Binary image with pixels in convex hull set to True.

    Notes
    -----
    This function uses skimage.morphology.label to define unique objects,
    finds the convex hull of each using convex_hull_image, and combines
    these regions with logical OR. Be aware the convex hulls of unconnected
    objects may overlap in the result. If this is suspected, consider using
    convex_hull_image separately on each object.

    """

    if neighbors != 4 and neighbors != 8:
        raise ValueError('Neighbors must be either 4 or 8.')

    labeled_im = label(image, neighbors, background=0)
    convex_obj = np.zeros(image.shape, dtype=bool)
    convex_img = np.zeros(image.shape, dtype=bool)

    for i in range(0, labeled_im.max() + 1):
        convex_obj = convex_hull_image(labeled_im == i)
        convex_img = np.logical_or(convex_img, convex_obj)

    return convex_img
