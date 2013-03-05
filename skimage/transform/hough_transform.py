__all__ = ['hough', 'hough_line', 'hough_circle', 'hough_peaks', 'probabilistic_hough']

from itertools import izip as zip

import numpy as np
from scipy import ndimage
from ._hough_transform import _probabilistic_hough
from skimage import measure, morphology


def _hough(img, theta=None):
    if img.ndim != 2:
        raise ValueError('The input image must be 2-D')

    if theta is None:
        theta = np.linspace(-np.pi / 2, np.pi / 2, 180)

    # compute the vertical bins (the distances)
    d = np.ceil(np.hypot(*img.shape))
    nr_bins = 2 * d
    bins = np.linspace(-d, d, nr_bins)

    # allocate the output image
    out = np.zeros((nr_bins, len(theta)), dtype=np.uint64)

    # precompute the sin and cos of the angles
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # find the indices of the non-zero values in
    # the input image
    y, x = np.nonzero(img)

    # x and y can be large, so we can't just broadcast to 2D
    # arrays as we may run out of memory. Instead we process
    # one vertical slice at a time.
    for i, (cT, sT) in enumerate(zip(cos_theta, sin_theta)):

        # compute the base distances
        distances = x * cT + y * sT

        # round the distances to the nearest integer
        # and shift them to a nonzero bin
        shifted = np.round(distances) - bins[0]

        # cast the shifted values to ints to use as indices
        indices = shifted.astype(np.int)

        # use bin count to accumulate the coefficients
        bincount = np.bincount(indices)

        # finally assign the proper values to the out array
        out[:len(bincount), i] = bincount

    return out, theta, bins

_py_hough = _hough

# try to import and use the faster Cython version if it exists
try:
    from ._hough_transform import _hough
except ImportError:
    pass


def probabilistic_hough(img, threshold=10, line_length=50, line_gap=10,
                        theta=None):
    """Return lines from a progressive probabilistic line Hough transform.

    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    threshold : int
        Threshold
    line_length : int, optional (default 50)
        Minimum accepted length of detected lines.
        Increase the parameter to extract longer lines.
    line_gap : int, optional, (default 10)
        Maximum gap between pixels to still form a line.
        Increase the parameter to merge broken lines more aggresively.
    theta : 1D ndarray, dtype=double, optional, default (-pi/2 .. pi/2)
        Angles at which to compute the transform, in radians.

    Returns
    -------
    lines : list
      List of lines identified, lines in format ((x0, y0), (x1, y0)), indicating
      line start and end.

    References
    ----------
    .. [1] C. Galamhos, J. Matas and J. Kittler, "Progressive probabilistic
           Hough transform for line detection", in IEEE Computer Society
           Conference on Computer Vision and Pattern Recognition, 1999.
    """
    return _probabilistic_hough(img, threshold, line_length, line_gap, theta)

from skimage._shared.utils import deprecated

@deprecated('hough_line')
def hough(img, theta=None):
    return hough_line(img, theta)

from ._hough_transform import _hough_circle

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
    theta : ndarray
        Angles at which the transform was computed.
    distances : ndarray
        Distance values.

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

    >>> out, angles, d = hough(img)

    .. plot:: hough_tf.py

    """
    return _hough(img, theta)

def hough_circle(img, radius, normalize=True):
    """Perform a circular Hough transform.

    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    radius : ndarray
        Radii at which to compute the Hough transform.
    normalize : boolean, optional
        Normalize the accumulator with the number
        of pixels used to draw the radius

    Returns
    -------
    H : 3D ndarray (radius index, (M, N) ndarray)
        Hough transform accumulator for each radius

    """
    return _hough_circle(img, radius.astype(np.intp), normalize)

def hough_peaks(hspace, angles, dists, min_distance=10, min_angle=10,
                threshold=None, num_peaks=np.inf):
    """Return peaks in hough transform.

    Identifies most prominent lines separated by a certain angle and distance in
    a hough transform. Non-maximum suppression with different sizes is applied
    separately in the first (distances) and second (angles) dimension of the
    hough space to identify peaks.

    Parameters
    ----------
    hspace : (N, M) array
        Hough space returned by the `hough` function.
    angles : (M,) array
        Angles returned by the `hough` function. Assumed to be continuous
        (`angles[-1] - angles[0] == PI`).
    dists : (N, ) array
        Distances returned by the `hough` function.
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
    >>> import numpy as np
    >>> from skimage.transform import hough, hough_peaks
    >>> from skimage.draw import line
    >>> img = np.zeros((15, 15), dtype=np.bool_)
    >>> rr, cc = line(0, 0, 14, 14)
    >>> img[rr, cc] = 1
    >>> rr, cc = line(0, 14, 14, 0)
    >>> img[cc, rr] = 1
    >>> hspace, angles, dists = hough(img)
    >>> hspace, angles, dists = hough_peaks(hspace, angles, dists)
    >>> angles
    array([  0.74590887,  -0.79856126])
    >>> dists
    array([  10.74418605,  0.51162791])

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

    label_hspace = morphology.label(hspace_t)
    props = measure.regionprops(label_hspace, ['Centroid'])
    coords = np.array([np.round(p['Centroid']) for p in props], dtype=int)

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
