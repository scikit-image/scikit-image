import numpy as np
from scipy import ndimage


def peak_local_max(image, min_distance=10, threshold=0.1):
    """Return coordinates of peaks in an image.

    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).

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

