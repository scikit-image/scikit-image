import numpy as np
from scipy.ndimage.filters import maximum_filter

from fast_cy import _corner_fast 


def corner_fast(image, n=12, threshold=0.15):

    """Extract FAST corners for a given image.

    Parameters
    ----------
    image : 2D ndarray
        Input image.
    n : int
        Number of consecutive pixels out of 16 pixels on the circle that
        should be brighter or darker with respect to test pixel above the
        `threshold` so as to classify the test pixel as a FAST corner. Also
        stands for the n in `FAST-n` corner detector.
    threshold : float
        Threshold used in deciding whether the pixels on the circle are
        brighter, darker or similar w.r.t. the test pixel. Decrease the
        threshold when more corners are desired and vice-versa.

    Returns
    -------
    corners : (N, 2) ndarray
        Location i.e. (row, col) of extracted FAST corners.

    References
    ----------
    .. [1] Edward Rosten and Tom Drummond
           "Machine Learning for high-speed corner detection",
           http://www.edwardrosten.com/work/rosten_2006_machine.pdf

    """
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("Only 2-D gray-scale images supported.")

    image = np.ascontiguousarray(image, dtype=np.double)
    corner_response = _corner_fast(image, n, threshold)

    # Non-maximal Suppression
    corner_zero_mask = corner_response != 0
    maximas = (maximum_filter(corner_response, (3, 3)) == corner_response) & corner_zero_mask
    x, y = np.where(maximas == True)

    corners = np.squeeze(np.dstack((x, y)))
    return corners
