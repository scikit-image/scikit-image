import numpy as np
from scipy import ndimage
from .. import measure, morphology
from ._hough_transform import (_hough_circle, _hough_ellipse,
                               _hough_line, _probabilistic_hough_line)


def hough_line_peaks(hspace, angles, dists, min_distance=9, min_angle=10,
                     threshold=None, num_peaks=np.inf):
    """Return peaks in Hough transform.

    Identifies most prominent lines separated by a certain angle and distance
    in a hough transform. Non-maximum suppression with different sizes is
    applied separately in the first (distances) and second (angles) dimension
    of the hough space to identify peaks.

    Parameters
    ----------
    hspace : (N, M) array
        Hough space returned by the `hough_line` function.
    angles : (M,) array
        Angles returned by the `hough_line` function. Assumed to be continuous.
        (`angles[-1] - angles[0] == PI`).
    dists : (N, ) array
        Distances returned by the `hough_line` function.
    min_distance : int
        Minimum distance separating lines (maximum filter size for first
        dimension of hough space).
    min_angle : int
        Minimum angle separating lines (maximum filter size for second
        dimension of hough space).
    threshold : float
        Minimum intensity of peaks. Default is `0.5 * max(hspace)`.
    num_peaks : int
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` coordinates based on peak intensity.

    Returns
    -------
    hspace, angles, dists : tuple of array
        Peak values in hough space, angles and distances.

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

    hspace = hspace.copy()
    rows, cols = hspace.shape

    if threshold is None:
        threshold = 0.5 * np.max(hspace)

    distance_size = 2 * min_distance + 1
    angle_size = 2 * min_angle + 1
    hspace_max = ndimage.maximum_filter1d(hspace, size=distance_size, axis=0,
                                          mode='constant', cval=0)
    hspace_max = ndimage.maximum_filter1d(hspace_max, size=angle_size, axis=1,
                                          mode='constant', cval=0)
    mask = (hspace == hspace_max)
    hspace *= mask
    hspace_t = hspace > threshold

    label_hspace = measure.label(hspace_t)
    props = measure.regionprops(label_hspace)
    coords = np.array([np.round(p.centroid) for p in props], dtype=int)

    hspace_peaks = []
    dist_peaks = []
    angle_peaks = []

    # relative coordinate grid for local neighbourhood suppression
    dist_ext, angle_ext = np.mgrid[-min_distance:min_distance + 1,
                                   -min_angle:min_angle + 1]

    for dist_idx, angle_idx in coords:
        accum = hspace[dist_idx, angle_idx]
        if accum > threshold:
            # absolute coordinate grid for local neighbourhood suppression
            dist_nh = dist_idx + dist_ext
            angle_nh = angle_idx + angle_ext

            # no reflection for distance neighbourhood
            dist_in = np.logical_and(dist_nh > 0, dist_nh < rows)
            dist_nh = dist_nh[dist_in]
            angle_nh = angle_nh[dist_in]

            # reflect angles and assume angles are continuous, e.g.
            # (..., 88, 89, -90, -89, ..., 89, -90, -89, ...)
            angle_low = angle_nh < 0
            dist_nh[angle_low] = rows - dist_nh[angle_low]
            angle_nh[angle_low] += cols
            angle_high = angle_nh >= cols
            dist_nh[angle_high] = rows - dist_nh[angle_high]
            angle_nh[angle_high] -= cols

            # suppress neighbourhood
            hspace[dist_nh, angle_nh] = 0

            # add current line to peaks
            hspace_peaks.append(accum)
            dist_peaks.append(dists[dist_idx])
            angle_peaks.append(angles[angle_idx])

    hspace_peaks = np.array(hspace_peaks)
    dist_peaks = np.array(dist_peaks)
    angle_peaks = np.array(angle_peaks)

    if num_peaks < len(hspace_peaks):
        idx_maxsort = np.argsort(hspace_peaks)[::-1][:num_peaks]
        hspace_peaks = hspace_peaks[idx_maxsort]
        dist_peaks = dist_peaks[idx_maxsort]
        angle_peaks = angle_peaks[idx_maxsort]

    return hspace_peaks, angle_peaks, dist_peaks


def hough_circle(image, radius, normalize=True, full_output=False):
    """Perform a circular Hough transform.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image with nonzero values representing edges.
    radius : ndarray
        Radii at which to compute the Hough transform.
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
    """
    return _hough_circle(image, radius.astype(np.intp),
                         normalize=normalize, full_output=full_output)

def hough_ellipse(img, threshold=4, accuracy=1, min_size=4, max_size=None):
    """Perform an elliptical Hough transform.

    Parameters
    ----------
    img : (M, N) ndarray
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
    result : ndarray with fields [(accumulator, y0, x0, a, b, orientation)]
          Where ``(yc, xc)`` is the center, ``(a, b)`` the major and minor
          axes, respectively. The `orientation` value follows
          `skimage.draw.ellipse_perimeter` convention.

    Examples
    --------
    >>> from skimage.draw import ellipse_perimeter
    >>> img = np.zeros((25, 25), dtype=np.uint8)
    >>> rr, cc = ellipse_perimeter(10, 10, 6, 8)
    >>> img[cc, rr] = 1
    >>> result = hough_ellipse(img, threshold=8)
    >>> result
    [(10, 10.0, 8.0, 6.0, 0.0, 10.0)]

    Notes
    -----
    The accuracy must be chosen to produce a peak in the accumulator
    distribution. In other words, a flat accumulator distribution with low
    values may be caused by a too low bin size.

    This function is a wrapper for Cython code.

    References
    ----------
    .. [1] Xie, Yonghong, and Qiang Ji. "A new efficient ellipse detection
           method." Pattern Recognition, 2002. Proceedings. 16th International
           Conference on. Vol. 2. IEEE, 2002
    """
    return _hough_ellipse(img, threshold, accuracy, min_size, max_size)

def hough_line(img, theta=None):
    """Perform a straight line Hough transform.

    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    theta : 1D ndarray of double
        Angles at which to compute the transform, in radians.
        Defaults to -pi/2 .. pi/2

    Returns
    -------
    H : 2-D ndarray of uint64
        Hough transform accumulator.
    theta : ndarray, optional
        Angles at which the transform was computed, in radians.
    distances : ndarray
        Distance values.

    Notes
    -----
    The origin is the top left corner of the original image.
    X and Y axis are horizontal and vertical edges respectively.
    The distance is the minimal algebraic distance from the origin
    to the detected line.

    Examples
    --------
    Generate a test image:

    >>> img = np.zeros((100, 150), dtype=bool)
    >>> img[30, :] = 1
    >>> img[:, 65] = 1
    >>> img[35:45, 35:50] = 1
    >>> for i in range(90):
    ...     img[i, i] = 1
    >>> img += np.random.random(img.shape) > 0.95

    Apply the Hough transform:

    >>> out, angles, d = hough_line(img)

    .. plot:: hough_tf.py

    """
    return _hough_line(img, theta)


def probabilistic_hough_line(img, threshold=10, line_length=50,
                             line_gap=10, theta=None):
    """Return lines from a progressive probabilistic line Hough transform.

    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    threshold : int, optional
        Threshold
    line_length : int, optional
        Minimum accepted length of detected lines.
        Increase the parameter to extract longer lines.
    line_gap : int, optional
        Maximum gap between pixels to still form a line.
        Increase the parameter to merge broken lines more aggresively.
    theta : 1D ndarray, dtype=double, optional
        Angles at which to compute the transform, in radians.
        If None, use a range from -pi/2 to pi/2.

    Returns
    -------
    lines : list
      List of lines identified, lines in format ((x0, y0), (x1, y0)),
      indicating line start and end.

    Notes
    -----
    This function is a wrapper for Cython code.

    References
    ----------
    .. [1] C. Galamhos, J. Matas and J. Kittler, "Progressive probabilistic
           Hough transform for line detection", in IEEE Computer Society
           Conference on Computer Vision and Pattern Recognition, 1999.
    """
    return _probabilistic_hough_line(img, threshold, line_length,
                                     line_gap, theta)
