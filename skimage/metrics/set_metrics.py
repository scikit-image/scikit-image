import warnings

import numpy as np
from scipy.spatial import cKDTree
from ..measure import find_contours


def _hausdorff_distance_set(a_points, b_points, method = 'standard'):
    """Calculate the Hausdorff distance between two sets of points.

    Parameters
    ----------
    a_points, b_points: ndarray
        Arrays containing the coordinates of the two sets of points.

    Returns
    -------
    distance : float
        The Hausdorff distance between the two sets of points ``a_points``
        and ``b_points``, using the Euclidean distance.
    """
    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    fwd, bwd = (
        cKDTree(a_points).query(b_points, k=1)[0],
        cKDTree(b_points).query(a_points, k=1)[0],
    )

    if method == 'standard':  # standard Hausdorff distance
        return max(max(fwd), max(bwd))
    elif method == 'modified':  # modified Hausdorff distance
        return max(np.mean(fwd), np.mean(bwd))


def _hausdorff_pair_set(a_points, b_points):
    """Finds the pair of points that are Hausdorff's Distance apart
    between two sets of points.

    Parameters
    ----------
    a_points, b_points: ndarray
        Arrays containing the coordinates of the two sets of points.

    """
    # If either of the sets are empty, there is no corresponding pair of points
    if len(a_points) == 0 or len(b_points) == 0:
        warnings.warn("One or both of the images is empty.", stacklevel=2)
        return (), ()

    nearest_dists_from_b, nearest_a_point_indices_from_b = cKDTree(a_points).query(
        b_points
    )
    nearest_dists_from_a, nearest_b_point_indices_from_a = cKDTree(b_points).query(
        a_points
    )

    max_index_from_a = nearest_dists_from_b.argmax()
    max_index_from_b = nearest_dists_from_a.argmax()

    max_dist_from_a = nearest_dists_from_b[max_index_from_a]
    max_dist_from_b = nearest_dists_from_a[max_index_from_b]

    if max_dist_from_b > max_dist_from_a:
        return (
            a_points[max_index_from_b],
            b_points[nearest_b_point_indices_from_a[max_index_from_b]],
        )
    else:
        return (
            a_points[nearest_a_point_indices_from_b[max_index_from_a]],
            b_points[max_index_from_a],
        )


def hausdorff_distance(image0, image1, method="standard"):
    """Calculate the Hausdorff distance between nonzero elements of given images.
    To use as a segmentation metric, the method should receive as input images
    containing the contours of the objects as nonzero elements. To use with
    segmentation masks as inputs, see the method ``hausdorff_distance_mask``.

    Parameters
    ----------
    image0, image1 : ndarray
        Arrays where ``True`` represents a point that is included in a
        set of points. Both arrays must have the same shape.
    method : {'standard', 'modified'}, optional, default = 'standard'
        The method to use for calculating the Hausdorff distance.
        ``standard`` is the standard Hausdorff distance, while ``modified``
        is the modified Hausdorff distance.
    Returns
    -------
    distance : float
        The Hausdorff distance between coordinates of nonzero pixels in
        ``image0`` and ``image1``, using the Euclidean distance.
    Notes
    -----
    The Hausdorff distance [1]_ is the maximum distance between any point on
    ``image0`` and its nearest point on ``image1``, and vice-versa.
    The Modified Hausdorff Distance (MHD) has been shown to perform better
    than the directed Hausdorff Distance (HD) in the following work by
    Dubuisson et al. [2]_. The function calculates forward and backward
    mean distances and returns the largest of the two.
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance
    .. [2] M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
       matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
       :DOI:`10.1109/ICPR.1994.576361`
       http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.8155
    Examples
    --------
    >>> points_a = (3, 0)
    >>> points_b = (6, 0)
    >>> shape = (7, 1)
    >>> image_a = np.zeros(shape, dtype=bool)
    >>> image_b = np.zeros(shape, dtype=bool)
    >>> image_a[points_a] = True
    >>> image_b[points_b] = True
    >>> hausdorff_distance(image_a, image_b)
    3.0

    """

    if method not in ('standard', 'modified'):
        raise ValueError(f'unrecognized method {method}')
    if image0.shape != image1.shape:
        raise ValueError(f'shape of image0 {image0.shape} and image1 {image1.shape} should be equal.')

    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))

    return _hausdorff_distance_set(a_points, b_points, method)


def hausdorff_distance_mask(image0, image1, method='standard'):
    """Calculate the Hausdorff distance between the contours of two segmentation masks.

    Parameters
    ----------
    image0, image1 : ndarray
        Arrays where ``True`` represents a pixel from a segmented object. Both arrays must have the same shape.
    method : {'standard', 'modified'}, optional, default = 'standard'
        The method to use for calculating the Hausdorff distance.
        ``standard`` is the standard Hausdorff distance, while ``modified``
        is the modified Hausdorff distance.
    Returns
    -------
    distance : float
        The Hausdorff distance between coordinates of the segmentation mask contours in
        ``image0`` and ``image1``, using the Euclidean distance.
    Notes
    -----
    The Hausdorff distance [1]_ is the maximum distance between any point on the
    contour of ``image0`` and its nearest point on the contour of ``image1``, and
    vice-versa.
    The Modified Hausdorff Distance (MHD) has been shown to perform better
    than the directed Hausdorff Distance (HD) in the following work by
    Dubuisson et al. [2]_. The function calculates forward and backward
    mean distances and returns the largest of the two.
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance
    .. [2] M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
       matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
       :DOI:`10.1109/ICPR.1994.576361`
       http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.8155
    Examples
    --------
    >>> ground_truth = np.zeros((10, 10), dtype=bool)
    >>> predicted = ground_truth.copy()
    >>> ground_truth[2:9, 2:9] = True
    >>> predicted[4:7, 2:9] = True
    >>> hausdorff_distance_mask(ground_truth, predicted)
    2.0
    """

    if method not in ('standard', 'modified'):
        raise ValueError(f'unrecognized method {method}')
    if image0.shape != image1.shape:
        raise ValueError(f'shape of image0 {image0.shape} and image1 {image1.shape} should be equal.')

    # Handle empty images
    if not np.any(image0):  # no nonzero elements in image0
        return 0.0 if not np.any(image1) else np.inf
    elif not np.any(image1):
        return np.inf

    a_points = np.concatenate(find_contours(image0>0))
    b_points = np.concatenate(find_contours(image1>0))

    return _hausdorff_distance_set(a_points, b_points, method)


def hausdorff_pair(image0, image1):
    """Returns pair of points that are Hausdorff distance apart between nonzero
    elements of given images.

    The Hausdorff distance [1]_ is the maximum distance between any point on
    ``image0`` and its nearest point on ``image1``, and vice-versa.

    Parameters
    ----------
    image0, image1 : ndarray
        Arrays where ``True`` represents a point that is included in a
        set of points. Both arrays must have the same shape.

    Returns
    -------
    point_a, point_b : array
        A pair of points that have Hausdorff distance between them.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance

    Examples
    --------
    >>> points_a = (3, 0)
    >>> points_b = (6, 0)
    >>> shape = (7, 1)
    >>> image_a = np.zeros(shape, dtype=bool)
    >>> image_b = np.zeros(shape, dtype=bool)
    >>> image_a[points_a] = True
    >>> image_b[points_b] = True
    >>> hausdorff_pair(image_a, image_b)
    (array([3, 0]), array([6, 0]))
    """
    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))

    return _hausdorff_pair_set(a_points, b_points)


def hausdorff_pair_mask(image0, image1):
    """Returns pair of points that are Hausdorff's Distance apart between
    the contours of two segmentation masks.

    The Hausdorff distance [1]_ is the maximum distance between any point on
    ``image0`` and its nearest point on ``image1``, and vice-versa.

    Parameters
    ----------
    image0, image1 : ndarray
        Arrays where ``True`` represents a point that is included in a
        set of points. Both arrays must have the same shape.

    Returns
    -------
    point_a, point_b : array
        A pair of points that have Hausdorff distance between them.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance

    Examples
    --------
    >>> ground_truth = np.zeros((10, 10), dtype=bool)
    >>> predicted = ground_truth.copy()
    >>> ground_truth[2:9, 2:9] = True
    >>> predicted[4:7, 2:9] = True
    >>> hausdorff_pair_mask(ground_truth, predicted)
    (array([8.5, 6.]), array([6.5, 6.]))
    """

    # Handle empty images
    if not np.any(image0) or not np.any(image1):
        warnings.warn("One or both of the images is empty.", stacklevel=2)
        return (), ()

    a_points = np.concatenate(find_contours(image0>0))
    b_points = np.concatenate(find_contours(image1>0))

    return _hausdorff_pair_set(a_points, b_points)
