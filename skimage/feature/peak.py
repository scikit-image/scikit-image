import numpy as np
from scipy import ndimage


def peak_min_dist(image, min_distance=10, threshold=0.1):
    """Return coordinates of peaks in an image.

    Candidate peaks are determined by a relative `threshold`, and peaks that
    are too close (as determined by `min_distance`) to larger peaks are
    rejected.

    Parameters
    ----------
    image: ndarray of floats
        Input image.

    min_distance: int, optional
        Minimum number of pixels separating peaks and image boundary.

    threshold: float, optional
        Candidate peaks are calculated as `max(image) * threshold`.

    Returns
    -------
    coordinates : (N, 2) array
        (row, column) coordinates of peaks.
    """
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

    # find top corner candidates above a threshold
    corner_threshold = np.max(image.ravel()) * threshold
    image_t = (image >= corner_threshold) * 1

    # get coordinates of peaks
    coordinates = np.transpose(image_t.nonzero())

    return coordinates

