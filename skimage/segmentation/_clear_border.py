import numpy as np
from scipy.ndimage import label


def clear_border(image, buffer_size=0, bgval=0):
    """Clear objects connected to image border.

    The changes will be applied to the input image.

    Parameters
    ----------
    image : (N, M) array
        Binary image.
    buffer_size : int, optional
        Define additional buffer around image border.
    bgval : float or int, optional
        Value for cleared objects.

    Returns
    -------
    image : (N, M) array
        Cleared binary image.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.segmentation import clear_border
    >>> image = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0],
    ...                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
    ...                   [1, 0, 0, 1, 0, 1, 0, 0, 0],
    ...                   [0, 0, 1, 1, 1, 1, 1, 0, 0],
    ...                   [0, 1, 1, 1, 1, 1, 1, 1, 0],
    ...                   [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> clear_border(image)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 1, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    """

    rows, cols = image.shape
    if buffer_size >= rows or buffer_size >= cols:
        raise ValueError("buffer size may not be greater than image size")

    # create borders with buffer_size
    borders = np.zeros_like(image, dtype=np.bool_)
    ext = buffer_size + 1
    borders[:ext] = True
    borders[- ext:] = True
    borders[:, :ext] = True
    borders[:, - ext:] = True

    labels, number = label(image)

    # determine all objects that are connected to borders
    borders_indices = np.unique(labels[borders])
    indices = np.arange(number + 1)
    # mask all label indices that are connected to borders
    label_mask = np.in1d(indices, borders_indices)
    # create mask for pixels to clear
    mask = label_mask[labels.ravel()].reshape(labels.shape)

    # clear border pixels
    image[mask] = bgval

    return image
