import warnings
import numpy as np
from scipy import ndimage


def peak_local_max(image, min_distance=10, threshold='deprecated',
                   threshold_abs=0, threshold_rel=0.1, num_peaks=np.inf):
    """Return coordinates of peaks in an image.

    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).

    NOTE: If peaks are flat (i.e. multiple pixels have exact same intensity),
    the coordinates of all pixels are returned.

    Parameters
    ----------
    image : ndarray of floats
        Input image.
    min_distance : int
        Minimum number of pixels separating peaks and image boundary.
    threshold : float
        Deprecated. See `threshold_rel`.
    threshold_abs : float
        Minimum intensity of peaks.
    threshold_rel : float
        Minimum intensity of peaks calculated as `max(image) * threshold_rel`.
    num_peaks : int
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` coordinates based on peak intensity.

    Returns
    -------
    coordinates : (N, 2) array
        (row, column) coordinates of peaks.

    Notes
    -----
    The peak local maximum function returns the coordinates of local peaks (maxima)
    in a image. A maximum filter is used for finding local maxima. This operation
    dilates the original image. After comparison between dilated and original image,
    peak_local_max function returns the coordinates of peaks where
    dilated image = original.

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
    if np.all(image == image.flat[0]):
        return []
    image = image.copy()
    # Non maximum filter
    size = 2 * min_distance + 1
    image_max = ndimage.maximum_filter(image, size=size, mode='constant')
    mask = (image == image_max)
    image *= mask

    # Remove the image borders
    image[:min_distance] = 0
    image[-min_distance:] = 0
    image[:, :min_distance] = 0
    image[:, -min_distance:] = 0

    if not threshold == 'deprecated':
        msg = "`threshold` parameter deprecated; use `threshold_rel instead."
        warnings.warn(msg, DeprecationWarning)
        threshold_rel = threshold
    # find top peak candidates above a threshold
    peak_threshold = max(np.max(image.ravel()) * threshold_rel, threshold_abs)
    image_t = (image > peak_threshold) * 1

    # get coordinates of peaks
    coordinates = np.transpose(image_t.nonzero())

    if coordinates.shape[0] > num_peaks:
        intensities = image[coordinates[:, 0], coordinates[:, 1]]
        idx_maxsort = np.argsort(intensities)[::-1]
        coordinates = coordinates[idx_maxsort][:num_peaks]

    return coordinates
