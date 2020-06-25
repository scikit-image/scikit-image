import numpy as np

from ._set_metrics import hausdorff_distance_onesided


def hausdorff_distance(image0, image1):
    """Calculate the Hausdorff distance between nonzero elements of given images.

    The Hausdorff distance [1]_ is the maximum distance between any point on
    ``image0`` and its nearest point on ``image1``, and vice-versa.

    Parameters
    ----------
    image0, image1 : ndarray
        Arrays where ``True`` represents a point that is included in a
        set of points. Both arrays must have the same shape.

    Returns
    -------
    distance : float
        The Hausdorff distance between coordinates of nonzero pixels in
        ``image0`` and ``image1``, using the Euclidian distance.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance

    Examples
    --------
    >>> points_a = (3, 0)
    >>> points_b = (6, 0)
    >>> shape = (7, 1)
    >>> image_a = np.zeros(shape, dtype=np.bool)
    >>> image_b = np.zeros(shape, dtype=np.bool)
    >>> image_a[points_a] = True
    >>> image_b[points_b] = True
    >>> hausdorff_distance(image_a, image_b)
    3.0

    """
    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))

    # Handle empty sets properly
    if len(a_points) == 0 or len(b_points) == 0:
        if len(a_points) == len(b_points):
            # Both sets are empty and thus the distance is zero
            return 0.
        else:
            # Exactly one set is empty; the distance is infinite
            return np.inf

    a_points = np.ascontiguousarray(a_points, dtype=np.float64)
    b_points = np.ascontiguousarray(b_points, dtype=np.float64)
    return max(hausdorff_distance_onesided(a_points, b_points),
               hausdorff_distance_onesided(b_points, a_points))
