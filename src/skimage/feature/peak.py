from textwrap import dedent

import numpy as np
import scipy.ndimage as ndi

from .. import measure
from ..util import PendingSkimage2Change
from .._shared._warnings import warn_external

import skimage2 as ski2


def peak_local_max(
    image,
    min_distance=1,
    threshold_abs=None,
    threshold_rel=None,
    exclude_border=True,
    num_peaks=None,
    footprint=None,
    labels=None,
    num_peaks_per_label=None,
    p_norm=np.inf,
):
    """Find peaks in an image as coordinate list.

    Peaks are the local maxima in a region of ``floor(2 * min_distance + 1)``
    (i.e. peaks are separated by at least `min_distance`).

    If both `threshold_abs` and `threshold_rel` are provided, the maximum
    of the two is chosen as the minimum intensity threshold of peaks.

    .. versionchanged:: 0.18
        Prior to version 0.18, peaks of the same height within a radius of
        `min_distance` were all returned, but this could cause unexpected
        behaviour. From 0.18 onwards, an arbitrary peak within the region is
        returned. See issue gh-2592.

    Parameters
    ----------
    image : ndarray
        Input image.
    min_distance : float, optional
        The minimal allowed distance separating peaks. To find the
        maximum number of peaks, use `min_distance=1`.
    threshold_abs : float, optional
        Minimum intensity of peaks. By default, the absolute threshold is
        the minimum intensity of the image.
    threshold_rel : float, optional
        Minimum intensity of peaks, calculated as
        ``max(image) * threshold_rel``.
    exclude_border : int or tuple of int(s) or bool, optional
        Control peak detection close to the border of `image`.

        ``True``
            Exclude peaks that are within ``floor(min_distance)`` of the border.
        ``False`` or ``0``
            Distance to border has no effect, all peaks are identified.
        positive integer
            Exclude peaks, that are within this given distance of the border.
        tuple of positive integers
            Same as for a single integer but with different distances for each
            respective dimension.

        The value of `p_norm` has no impact on this border distance.
    num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` peaks based on highest peak intensity.

        .. deprecated:: 0.27
            Passing ``numpy.inf`` is deprecated,
            use the equivalent ``None`` instead.

    footprint : ndarray of dtype bool, optional
        Binary mask that determines the neighborhood (where ``True``) in which
        a peak must be a local maximum (see *Notes*). If not given, defaults to
        an array of ones of size ``floor(2 * min_distance + 1)``.
    labels : ndarray of dtype int, optional
        If provided, each unique region `labels == value` represents a unique
        region to search for peaks. Zero is reserved for background.
    num_peaks_per_label : int, optional
        Maximum number of peaks for each label.

        .. deprecated:: 0.27
            Passing ``numpy.inf`` is deprecated,
            use the equivalent ``None`` instead.

    p_norm : float, optional
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.  See also :func:`numpy.linalg.norm`.

    Returns
    -------
    output : ndarray of dtype int
        The coordinates of the peaks.

    Notes
    -----
    The peak local maximum function returns the coordinates of local peaks
    (maxima) in an image. Internally, a maximum filter is used for finding
    local maxima. This operation dilates the original image. After comparison
    of the dilated and original images, this function returns the coordinates
    of the peaks where the dilated image equals the original image.

    Examples
    --------
    >>> img1 = np.zeros((7, 7))
    >>> img1[3, 4] = 1
    >>> img1[3, 2] = 1.5
    >>> img1
    array([[0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 1.5, 0. , 1. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ]])

    >>> peak_local_max(img1, min_distance=1)
    array([[3, 2],
           [3, 4]])

    >>> peak_local_max(img1, min_distance=2)
    array([[3, 2]])

    >>> img2 = np.zeros((20, 20, 20))
    >>> img2[10, 10, 10] = 1
    >>> img2[15, 15, 15] = 1
    >>> peak_idx = peak_local_max(img2, exclude_border=0)
    >>> peak_idx
    array([[10, 10, 10],
           [15, 15, 15]])

    >>> peak_mask = np.zeros_like(img2, dtype=bool)
    >>> peak_mask[tuple(peak_idx.T)] = True
    >>> np.argwhere(peak_mask)
    array([[10, 10, 10],
           [15, 15, 15]])

    """
    warn_external(
        dedent("""\
        `skimage.feature.peak_local_max` is deprecated in favor of
        `skimage2.feature.peak_local_max` with new behavior:

        * Parameter `p_norm` defaults to 2 (Euclidean distance),
          was `numpy.inf` (Chebyshev distance)
        * Parameter `exclude_border` defaults to 1, was `True`
        * Parameter `exclude_border` no longer accepts `False` and `True`,
          pass 0 instead of `False`, or `min_distance` instead of `True`
        * Parameters after `image` are keyword-only

        To keep the old behavior when switching to `skimage2`, update your call
        according to the following cases:

        * `exclude_border` not passed, use `exclude_border=<value_of_min_distance>`
        * `exclude_border=True`, same as above
        * `exclude_border=False`, use `exclude_border=0`
        * `exclude_border=<int>`, no change necessary
        * `p_norm` not passed, use `p_norm=numpy.inf`
        * `p_norm=<float>, no change necessary

        Other keyword parameters can be left unchanged.
        """),
        category=PendingSkimage2Change,
    )

    # Deprecate passing `np.inf` to `num_peaks` and `num_peaks_per_label`
    if num_peaks is not None and np.isinf(num_peaks):
        num_peaks = None
        warn_external(
            "Passing `np.inf` to `num_peaks` is deprecated in version 0.27, "
            "use `num_peaks=None` instead",
            category=FutureWarning,
        )
    if num_peaks_per_label is not None and np.isinf(num_peaks_per_label):
        num_peaks_per_label = None
        warn_external(
            "Passing `np.inf` to `num_peaks_per_label` is deprecated in version 0.27, "
            "use `num_peaks_per_label=None` instead",
            category=FutureWarning,
        )

    if exclude_border is False:
        exclude_border = 0
    elif exclude_border is True:
        exclude_border = int(np.floor(min_distance))

    coordinates = ski2.feature.peak_local_max(
        image,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
        num_peaks=num_peaks,
        footprint=footprint,
        labels=labels,
        num_peaks_per_label=num_peaks_per_label,
        p_norm=p_norm,
    )
    return coordinates


def _prominent_peaks(
    image, min_xdistance=1, min_ydistance=1, threshold=None, num_peaks=np.inf
):
    """Return peaks with non-maximum suppression.

    Identifies most prominent features separated by certain distances.
    Non-maximum suppression with different sizes is applied separately
    in the first and second dimension of the image to identify peaks.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    min_xdistance : int
        Minimum distance separating features in the x dimension.
    min_ydistance : int
        Minimum distance separating features in the y dimension.
    threshold : float
        Minimum intensity of peaks. Default is `0.5 * max(image)`.
    num_peaks : int
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` coordinates based on peak intensity.

    Returns
    -------
    intensity, xcoords, ycoords : tuple of array
        Peak intensity values, x and y indices.
    """

    img = image.copy()
    rows, cols = img.shape

    if threshold is None:
        threshold = 0.5 * np.max(img)

    ycoords_size = 2 * min_ydistance + 1
    xcoords_size = 2 * min_xdistance + 1
    img_max = ndi.maximum_filter1d(
        img, size=ycoords_size, axis=0, mode='constant', cval=0
    )
    img_max = ndi.maximum_filter1d(
        img_max, size=xcoords_size, axis=1, mode='constant', cval=0
    )
    mask = img == img_max
    img *= mask
    img_t = img > threshold

    label_img = measure.label(img_t)
    props = measure.regionprops(label_img, img_max)

    # Sort the list of peaks by intensity, not left-right, so larger peaks
    # in Hough space cannot be arbitrarily suppressed by smaller neighbors
    props = sorted(props, key=lambda x: x.intensity_max)[::-1]
    coords = np.array([np.round(p.centroid) for p in props], dtype=int)

    img_peaks = []
    ycoords_peaks = []
    xcoords_peaks = []

    # relative coordinate grid for local neighborhood suppression
    ycoords_ext, xcoords_ext = np.mgrid[
        -min_ydistance : min_ydistance + 1, -min_xdistance : min_xdistance + 1
    ]

    for ycoords_idx, xcoords_idx in coords:
        accum = img_max[ycoords_idx, xcoords_idx]
        if accum > threshold:
            # absolute coordinate grid for local neighborhood suppression
            ycoords_nh = ycoords_idx + ycoords_ext
            xcoords_nh = xcoords_idx + xcoords_ext

            # no reflection for distance neighborhood
            ycoords_in = np.logical_and(ycoords_nh > 0, ycoords_nh < rows)
            ycoords_nh = ycoords_nh[ycoords_in]
            xcoords_nh = xcoords_nh[ycoords_in]

            # reflect xcoords and assume xcoords are continuous,
            # e.g. for angles:
            # (..., 88, 89, -90, -89, ..., 89, -90, -89, ...)
            xcoords_low = xcoords_nh < 0
            ycoords_nh[xcoords_low] = rows - ycoords_nh[xcoords_low]
            xcoords_nh[xcoords_low] += cols
            xcoords_high = xcoords_nh >= cols
            ycoords_nh[xcoords_high] = rows - ycoords_nh[xcoords_high]
            xcoords_nh[xcoords_high] -= cols

            # suppress neighborhood
            img_max[ycoords_nh, xcoords_nh] = 0

            # add current feature to peaks
            img_peaks.append(accum)
            ycoords_peaks.append(ycoords_idx)
            xcoords_peaks.append(xcoords_idx)

    img_peaks = np.array(img_peaks)
    ycoords_peaks = np.array(ycoords_peaks)
    xcoords_peaks = np.array(xcoords_peaks)

    if num_peaks < len(img_peaks):
        idx_maxsort = np.argsort(img_peaks)[::-1][:num_peaks]
        img_peaks = img_peaks[idx_maxsort]
        ycoords_peaks = ycoords_peaks[idx_maxsort]
        xcoords_peaks = xcoords_peaks[idx_maxsort]

    return img_peaks, xcoords_peaks, ycoords_peaks
