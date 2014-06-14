import numpy as np
from scipy import ndimage
from skimage import measure, morphology


def hough_line_peaks(hspace, angles, dists, min_distance=9, min_angle=10,
                     threshold=None, num_peaks=np.inf):
    """Return peaks in hough transform.

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
    >>> import numpy as np
    >>> from skimage.transform import hough_line, hough_peaks
    >>> from skimage.draw import line
    >>> img = np.zeros((15, 15), dtype=np.bool_)
    >>> rr, cc = line(0, 0, 14, 14)
    >>> img[rr, cc] = 1
    >>> rr, cc = line(0, 14, 14, 0)
    >>> img[cc, rr] = 1
    >>> hspace, angles, dists = hough_line(img)
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
