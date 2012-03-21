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
        
    Examples
    --------
    >>> square = np.zeros([10,10])
    >>> square[2:8,2:8]=1
    >>> square[3:7,3:7]=0
    >>> square
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    
    >>> image_max = ndimage.maximum_filter(square, size=3, 
    mode='constant')
    
    image_max is computed in peak_local_max function to enable
    the calculation of coordinates of peaks. It is the dilation of 
    square
    
    >>> image_max
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  0.,  0.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  0.,  0.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    
    After comparison between image_max and square, peak_local_max
    function returns the coordinates of peaks where
    square = image_max
    
    >>> peak_local_max(square, min_distance=1)
    array([[2, 2],
           [2, 3],
           [2, 4],
           [2, 5],
           [2, 6],
           [2, 7],
           [3, 2],
           [3, 7],
           [4, 2],
           [4, 7],
           [5, 2],
           [5, 7],
           [6, 2],
           [6, 7],
           [7, 2],
           [7, 3],
           [7, 4],
           [7, 5],
           [7, 6],
           [7, 7]])
    
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

