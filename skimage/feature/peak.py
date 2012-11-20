import numpy as np
import scipy.ndimage as ndi

def peak_local_max(image, min_distance=10, threshold_abs=0, threshold_rel=0.1,
                   exclude_border=True, indices=True, num_peaks=np.inf,
                   footprint=None, labels=None, **kwargs):
    """
    Find peaks in an image, and return them as coordinates or a boolean array.

    Peaks are the local maxima

    NOTE: If peaks are flat (i.e. multiple pixels have exact same intensity),
    the coordinates of all pixels are returned.

    Parameters
    ----------
    image : ndarray of floats
        Input image.

    min_distance : int, default 10.
        Minimum number of pixels separating peaks in a region of `2 *
        min_distance + 1` (i.e. peaks are separated by at least
        `min_distance`).
        If `exclude_border` is True, this value also excludes a border
        `min_distance` from the image boundary.
        To find the maximum number of points, use `min_distance=1`.

    threshold_abs : float, default 0.
        Minimum intensity of peaks.

    threshold_rel : float, default 0.1
        Minimum intensity of peaks calculated as `max(image) * threshold_rel`.

    exclude_border : bool, default True
        If True, `min_distance` excludes peaks from the border of the image as
        well as from each other.

    indices : bool, default True
        If True, the output will be a matrix representing peak coordinates.
        If False, the output will be a boolean matrix shaped as `image.shape`
            with peaks present at True elements.

    num_peaks : int, default np.inf
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` peaks based on highest peak intensity.

    footprint : ndarray of bools, optional
        If provided, `footprint == 1` represents the local region within which
        to search for peaks at every point in `image`.
        Overrides `min_distance`, except for border exclusion if
            `exclude_border` is True.

    labels : ndarray of ints, optional
        If provided, each unique region `labels == value` represents a unique
        region to search for peaks. Zero is reserved for background.

    threshold : float, optional
        Deprecated. If provided as a kwarg, will override `threshold_rel`.
        See `threshold_rel`.

    Returns
    -------
    output : (N, 2) array or ndarray of bools
        If `exclude_border = True`  : (row, column) coordinates of peaks.
        If `exclude_border = False` : Boolean array shaped like `image`,
            with peaks represented by True values.

    Notes
    -----
    The peak local maximum function returns the coordinates of local peaks
    (maxima) in a image. A maximum filter is used for finding local maxima.
    This operation dilates the original image. After comparison between
    dilated and original image, peak_local_max function returns the
    coordinates of peaks where dilated image = original.

    Examples
    --------
    >>> im = np.zeros((7, 7))
    >>> im[3, 4] = 1
    >>> im[3, 2] = 1.5
    >>> im
    array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  1.5,  0. ,  1. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])

    >>> peak_local_max(im, min_distance=1)
    array([[3, 2],
           [3, 4]])

    >>> peak_local_max(im, min_distance=2)
    array([[3, 2]])

    """
    # In the case of labels, recursively build and return an output
    # operating on each label separately; for API compatibility with
    # ..watershed.is_local_maximum()
    if labels is not None:
        label_values = np.unique(labels)
        # Reorder label values to have consecutive integers (no gaps)
        if np.any(np.diff(label_values) != 1):
            mask = labels >= 0
            labels[mask] = rank_order(labels[mask])[0].astype(labels.dtype)
        labels = labels.astype(np.int32)

        out = np.zeros_like(image)
        for label in labels:
            out += peak_local_max(image, min_distance=min_distance,
                                   threshold_abs=threshold_abs,
                                   threshold_rel=threshold_rel,
                                   exclude_border=exclude_border,
                                   indices=False, num_peaks=np.inf,
                                   footprint=footprint, labels=None,
                                   **kwargs)

        if indices is True:
            return np.transpose(out.nonzero())
        else:
            return out.astype(bool)


    if np.all(image == image.flat[0]):
        if indices is True:
            return []
        else:
            return np.zeros_like(image)

    image = image.copy()
    # Non maximum filter
    if footprint is not None:
        image_max = ndi.maximum_filter(image, footprint=footprint,
                                       mode='constant')
    else:
        size = 2 * min_distance + 1
        image_max = ndi.maximum_filter(image, size=size, mode='constant')
    mask = (image == image_max)
    image *= mask

    if exclude_border:
        # Remove the image borders
        image[:min_distance] = 0
        image[-min_distance:] = 0
        image[:, :min_distance] = 0
        image[:, -min_distance:] = 0

    if kwargs.has_key('threshold'):
        threshold_rel = kwargs['threshold']

    # find top peak candidates above a threshold
    peak_threshold = max(np.max(image.ravel()) * threshold_rel, threshold_abs)
    image_t = (image > peak_threshold) * 1

    # get coordinates of peaks
    coordinates = np.transpose(image_t.nonzero())

    if coordinates.shape[0] > num_peaks:
        intensities = image[coordinates[:, 0], coordinates[:, 1]]
        idx_maxsort = np.argsort(intensities)[::-1]
        coordinates = coordinates[idx_maxsort][:num_peaks]

    if indices is True:
        return coordinates
    else:
        out = np.zeros_like(image, dtype=bool)
        out[coordinates[:, 0], coordinates[:, 1]] = True
        return out
