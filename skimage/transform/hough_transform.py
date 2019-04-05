import numpy as np
from ._hough_transform import _hough_circle, _hough_ellipse
from ..transform_xy import (hough_line, hough_circle_peaks,
                            probabilistic_hough_line)


def hough_line_peaks(hspace, angles, dists, min_distance=9, min_angle=10,
                     threshold=None, num_peaks=np.inf):
    """Return peaks in a straight line Hough transform.

    Identifies most prominent lines separated by a certain angle and distance
    in a Hough transform. Non-maximum suppression with different sizes is
    applied separately in the first (distances) and second (angles) dimension
    of the Hough space to identify peaks.

    Parameters
    ----------
    hspace : (N, M) array
        Hough space returned by the `hough_line` function.
    angles : (M,) array
        Angles returned by the `hough_line` function. Assumed to be continuous.
        (`angles[-1] - angles[0] == PI`).
    dists : (N, ) array
        Distances returned by the `hough_line` function.
    min_distance : int, optional
        Minimum distance separating lines (maximum filter size for first
        dimension of hough space).
    min_angle : int, optional
        Minimum angle separating lines (maximum filter size for second
        dimension of hough space).
    threshold : float, optional
        Minimum intensity of peaks. Default is `0.5 * max(hspace)`.
    num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` coordinates based on peak intensity.

    Returns
    -------
    accum, angles, dists : tuple of array
        Peak values in Hough space, angles and distances.

    Examples
    --------
    >>> from skimage.transform import hough_line, hough_line_peaks
    >>> from skimage.draw import line
    >>> img = np.zeros((15, 15), dtype=np.bool_)
    >>> rr, cc = line(0, 0, 14, 14)
    >>> img[rr, cc] = 1
    >>> rr, cc = line(0, 14, 14, 0)
    >>> img[cc, rr] = 1
    >>> hspace, angles, dists = hough_line(img)
    >>> hspace, angles, dists = hough_line_peaks(hspace, angles, dists)
    >>> len(angles)
    2

    """
    from ..feature.peak import _prominent_peaks

    h, a, d = _prominent_peaks(hspace, min_xdistance=min_angle,
                               min_ydistance=min_distance,
                               threshold=threshold,
                               num_peaks=num_peaks)
    if a.any():
        return (h, angles[a], dists[d])
    else:
        return (h, np.array([]), np.array([]))


def hough_circle(image, radius, normalize=True, full_output=False):
    """Perform a circular Hough transform.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image with nonzero values representing edges.
    radius : scalar or sequence of scalars
        Radii at which to compute the Hough transform.
        Floats are converted to integers.
    normalize : boolean, optional (default True)
        Normalize the accumulator with the number
        of pixels used to draw the radius.
    full_output : boolean, optional (default False)
        Extend the output size by twice the largest
        radius in order to detect centers outside the
        input picture.

    Returns
    -------
    H : 3D ndarray (radius index, (M + 2R, N + 2R) ndarray)
        Hough transform accumulator for each radius.
        R designates the larger radius if full_output is True.
        Otherwise, R = 0.

    Examples
    --------
    >>> from skimage.transform import hough_circle
    >>> from skimage.draw import circle_perimeter
    >>> img = np.zeros((100, 100), dtype=np.bool_)
    >>> rr, cc = circle_perimeter(25, 35, 23)
    >>> img[rr, cc] = 1
    >>> try_radii = np.arange(5, 50)
    >>> res = hough_circle(img, try_radii)
    >>> ridx, r, c = np.unravel_index(np.argmax(res), res.shape)
    >>> r, c, try_radii[ridx]
    (25, 35, 23)

    """
    radius = np.atleast_1d(np.asarray(radius))
    return _hough_circle(image, radius.astype(np.intp),
                         normalize=normalize, full_output=full_output)


def hough_ellipse(image, threshold=4, accuracy=1, min_size=4, max_size=None):
    """Perform an elliptical Hough transform.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image with nonzero values representing edges.
    threshold: int, optional
        Accumulator threshold value.
    accuracy : double, optional
        Bin size on the minor axis used in the accumulator.
    min_size : int, optional
        Minimal major axis length.
    max_size : int, optional
        Maximal minor axis length.
        If None, the value is set to the half of the smaller
        image dimension.

    Returns
    -------
    result : ndarray with fields [(accumulator, rc, cc, a, b, orientation)].
          Where ``(rc, cc)`` is the center, ``(a, b)`` the major and minor
          axes, respectively. The `orientation` value follows
          `skimage.draw.ellipse_perimeter` convention.

    Examples
    --------
    >>> from skimage.transform import hough_ellipse
    >>> from skimage.draw import ellipse_perimeter
    >>> img = np.zeros((25, 25), dtype=np.uint8)
    >>> rr, cc = ellipse_perimeter(10, 10, 6, 8)
    >>> img[cc, rr] = 1
    >>> result = hough_ellipse(img, threshold=8)
    >>> result.tolist()
    [(10, 10.0, 10.0, 8.0, 6.0, 0.0)]

    Notes
    -----
    The accuracy must be chosen to produce a peak in the accumulator
    distribution. In other words, a flat accumulator distribution with low
    values may be caused by a too low bin size.

    References
    ----------
    .. [1] Xie, Yonghong, and Qiang Ji. "A new efficient ellipse detection
           method." Pattern Recognition, 2002. Proceedings. 16th International
           Conference on. Vol. 2. IEEE, 2002
    """
    return _hough_ellipse(image, threshold=threshold, accuracy=accuracy,
                          min_size=min_size, max_size=max_size)
