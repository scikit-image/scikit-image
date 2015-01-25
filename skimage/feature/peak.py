import numpy as np
import scipy.ndimage as ndi
from ..filters import rank_order


def peak_local_max(image, min_distance=10, threshold_abs=0, threshold_rel=0.1,
                   exclude_border=True, indices=True, num_peaks=np.inf,
                   footprint=None, labels=None):
    """
    Find peaks in an image, and return them as coordinates or a boolean array.

    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).

    NOTE: If peaks are flat (i.e. multiple adjacent pixels have identical
    intensities), the coordinates of all such pixels are returned.

    Parameters
    ----------
    image : ndarray of floats
        Input image.
    min_distance : int
        Minimum number of pixels separating peaks in a region of `2 *
        min_distance + 1` (i.e. peaks are separated by at least
        `min_distance`). If `exclude_border` is True, this value also excludes
        a border `min_distance` from the image boundary.
        To find the maximum number of peaks, use `min_distance=1`.
    threshold_abs : float
        Minimum intensity of peaks.
    threshold_rel : float
        Minimum intensity of peaks calculated as `max(image) * threshold_rel`.
    exclude_border : bool
        If True, `min_distance` excludes peaks from the border of the image as
        well as from each other.
    indices : bool
        If True, the output will be an array representing peak coordinates.
        If False, the output will be a boolean array shaped as `image.shape`
        with peaks present at True elements.
    num_peaks : int
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` peaks based on highest peak intensity.
    footprint : ndarray of bools, optional
        If provided, `footprint == 1` represents the local region within which
        to search for peaks at every point in `image`.  Overrides
        `min_distance`, except for border exclusion if `exclude_border=True`.
    labels : ndarray of ints, optional
        If provided, each unique region `labels == value` represents a unique
        region to search for peaks. Zero is reserved for background.

    Returns
    -------
    output : ndarray or ndarray of bools

        * If `indices = True`  : (row, column, ...) coordinates of peaks.
        * If `indices = False` : Boolean array shaped like `image`, with peaks
          represented by True values.

    Notes
    -----
    The peak local maximum function returns the coordinates of local peaks
    (maxima) in a image. A maximum filter is used for finding local maxima.
    This operation dilates the original image. After comparison between
    dilated and original image, peak_local_max function returns the
    coordinates of peaks where dilated image = original.

    Examples
    --------
    >>> img1 = np.zeros((7, 7))
    >>> img1[3, 4] = 1
    >>> img1[3, 2] = 1.5
    >>> img1
    array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  1.5,  0. ,  1. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])

    >>> peak_local_max(img1, min_distance=1)
    array([[3, 2],
           [3, 4]])

    >>> peak_local_max(img1, min_distance=2)
    array([[3, 2]])

    >>> img2 = np.zeros((20, 20, 20))
    >>> img2[10, 10, 10] = 1
    >>> peak_local_max(img2, exclude_border=False)
    array([[10, 10, 10]])

    """
    out = np.zeros_like(image, dtype=np.bool)
    # In the case of labels, recursively build and return an output
    # operating on each label separately
    if labels is not None:
        label_values = np.unique(labels)
        # Reorder label values to have consecutive integers (no gaps)
        if np.any(np.diff(label_values) != 1):
            mask = labels >= 1
            labels[mask] = 1 + rank_order(labels[mask])[0].astype(labels.dtype)
        labels = labels.astype(np.int32)

        # New values for new ordering
        label_values = np.unique(labels)
        for label in label_values[label_values != 0]:
            maskim = (labels == label)
            out += peak_local_max(image * maskim, min_distance=min_distance,
                                  threshold_abs=threshold_abs,
                                  threshold_rel=threshold_rel,
                                  exclude_border=exclude_border,
                                  indices=False, num_peaks=np.inf,
                                  footprint=footprint, labels=None)

        if indices is True:
            return np.transpose(out.nonzero())
        else:
            return out.astype(np.bool)

    if np.all(image == image.flat[0]):
        if indices is True:
            return []
        else:
            return out

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
        # zero out the image borders
        for i in range(image.ndim):
            image = image.swapaxes(0, i)
            image[:min_distance] = 0
            image[-min_distance:] = 0
            image = image.swapaxes(0, i)

    # find top peak candidates above a threshold
    peak_threshold = max(np.max(image.ravel()) * threshold_rel, threshold_abs)

    # get coordinates of peaks
    coordinates = np.argwhere(image > peak_threshold)

    if coordinates.shape[0] > num_peaks:
        intensities = image.flat[np.ravel_multi_index(coordinates.transpose(),image.shape)]
        idx_maxsort = np.argsort(intensities)[::-1]
        coordinates = coordinates[idx_maxsort][:num_peaks]

    if indices is True:
        return coordinates
    else:
        nd_indices = tuple(coordinates.T)
        out[nd_indices] = True
        return out
