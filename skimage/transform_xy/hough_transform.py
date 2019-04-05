import numpy as np
from ._hough_transform import (_hough_line,
                               _probabilistic_hough_line as _prob_hough_line)


def hough_line(image, theta=None):
    """Perform a straight line Hough transform.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image with nonzero values representing edges.
    theta : 1D ndarray of double, optional
        Angles at which to compute the transform, in radians.
        Defaults to a vector of 180 angles evenly spaced from -pi/2 to pi/2.

    Returns
    -------
    hspace : 2-D ndarray of uint64
        Hough transform accumulator.
    angles : ndarray
        Angles at which the transform is computed, in radians.
    distances : ndarray
        Distance values.

    Notes
    -----
    The origin is the top left corner of the original image.
    X and Y axis are horizontal and vertical edges respectively.
    The distance is the minimal algebraic distance from the origin
    to the detected line.
    The angle accuracy can be improved by decreasing the step size in
    the `theta` array.

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
    if image.ndim != 2:
        raise ValueError('The input image `image` must be 2D.')

    if theta is None:
        # These values are approximations of pi/2
        theta = np.linspace(-np.pi / 2, np.pi / 2, 180)

    return _hough_line(image, theta=theta)


def probabilistic_hough_line(image, threshold=10, line_length=50, line_gap=10,
                             theta=None, seed=None):
    """Return lines from a progressive probabilistic line Hough transform.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image with nonzero values representing edges.
    threshold : int, optional
        Threshold
    line_length : int, optional
        Minimum accepted length of detected lines.
        Increase the parameter to extract longer lines.
    line_gap : int, optional
        Maximum gap between pixels to still form a line.
        Increase the parameter to merge broken lines more aggressively.
    theta : 1D ndarray, dtype=double, optional
        Angles at which to compute the transform, in radians.
        If None, use a range from -pi/2 to pi/2.
    seed : int, optional
        Seed to initialize the random number generator.

    Returns
    -------
    lines : list
      List of lines identified, lines in format ((x0, y0), (x1, y1)),
      indicating line start and end.

    References
    ----------
    .. [1] C. Galamhos, J. Matas and J. Kittler, "Progressive probabilistic
           Hough transform for line detection", in IEEE Computer Society
           Conference on Computer Vision and Pattern Recognition, 1999.
    """

    if image.ndim != 2:
        raise ValueError('The input image `image` must be 2D.')

    if theta is None:
        theta = np.pi / 2 - np.arange(180) / 180.0 * np.pi

    return _prob_hough_line(image, threshold=threshold, line_length=line_length,
                            line_gap=line_gap, theta=theta, seed=seed)


def hough_circle_peaks(hspaces, radii, min_xdistance=1, min_ydistance=1,
                       threshold=None, num_peaks=np.inf,
                       total_num_peaks=np.inf, normalize=False):
    """Return peaks in a circle Hough transform.

    Identifies most prominent circles separated by certain distances in a
    Hough space. Non-maximum suppression with different sizes is applied
    separately in the first and second dimension of the Hough space to
    identify peaks.

    Parameters
    ----------
    hspaces : (N, M) array
        Hough spaces returned by the `hough_circle` function.
    radii : (M,) array
        Radii corresponding to Hough spaces.
    min_xdistance : int, optional
        Minimum distance separating centers in the x dimension.
    min_ydistance : int, optional
        Minimum distance separating centers in the y dimension.
    threshold : float, optional
        Minimum intensity of peaks in each Hough space.
        Default is `0.5 * max(hspace)`.
    num_peaks : int, optional
        Maximum number of peaks in each Hough space. When the
        number of peaks exceeds `num_peaks`, only `num_peaks`
        coordinates based on peak intensity are considered for the
        corresponding radius.
    total_num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` coordinates based on peak intensity.
    normalize : bool, optional
        If True, normalize the accumulator by the radius to sort the prominent
        peaks.

    Returns
    -------
    accum, cx, cy, rad : tuple of array
        Peak values in Hough space, x and y center coordinates and radii.

    Examples
    --------
    >>> from skimage import transform, draw
    >>> img = np.zeros((120, 100), dtype=int)
    >>> radius, x_0, y_0 = (20, 99, 50)
    >>> y, x = draw.circle_perimeter(y_0, x_0, radius)
    >>> img[x, y] = 1
    >>> hspaces = transform.hough_circle(img, radius)
    >>> accum, cx, cy, rad = hough_circle_peaks(hspaces, [radius,])
    """
    from ..feature.peak import _prominent_peaks

    r = []
    cx = []
    cy = []
    accum = []

    for rad, hp in zip(radii, hspaces):
        h_p, x_p, y_p = _prominent_peaks(hp,
                                         min_xdistance=min_xdistance,
                                         min_ydistance=min_ydistance,
                                         threshold=threshold,
                                         num_peaks=num_peaks)
        r.extend((rad,)*len(h_p))
        cx.extend(x_p)
        cy.extend(y_p)
        accum.extend(h_p)

    r = np.array(r)
    cx = np.array(cx)
    cy = np.array(cy)
    accum = np.array(accum)
    if normalize:
        s = np.argsort(accum / r)
    else:
        s = np.argsort(accum)

    if total_num_peaks != np.inf:
        tnp = total_num_peaks
        return (accum[s][::-1][:tnp], cx[s][::-1][:tnp], cy[s][::-1][:tnp],
                r[s][::-1][:tnp])

    return (accum[s][::-1], cx[s][::-1], cy[s][::-1], r[s][::-1])
