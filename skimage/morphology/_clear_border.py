import numpy as np
from skimage.measure import regionprops
from skimage.morphology import label


def clear_border(image, buffer_size=0, bgval=0):
    """Clear objects connected to image border.

    The changes will be applied to the input image.

    Parameters
    ----------
    image : (N, M) array
        binary image
    buffer_size : int, optional
        define additional buffer around image border
    bgval : float or int, optional
        value for cleared objects

    Returns
    -------
    image : (N, M) array
        cleared binary image
    """
    rows, cols = image.shape
    for prop in regionprops(label(image), ['BoundingBox', 'Image']):
        minr, minc, maxr, maxc = prop['BoundingBox']
        if (
            minr <= buffer_size
            or minc <= buffer_size
            or maxr >= rows - buffer_size
            or maxc >= cols - buffer_size
        ):
            r, c = np.nonzero(prop['Image'])
            image[minr + r, minc + c] = bgval
    return image
