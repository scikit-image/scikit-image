import numpy as np
#from scipy.ndimage import label
from ..measure import label


def clear_border(labels, buffer_size=0, bgval=0):
    """Clear objects connected to the label image border.

    The changes will be applied directly to the input.

    Parameters
    ----------
    labels : (N, M) array of int
        Label or binary image.
    buffer_size : int, optional
        The width of the border examined.  By default, only objects
        that touch the outside of the image are removed.
    bgval : float or int, optional
        Cleared objects are set to this value.

    Returns
    -------
    labels : (N, M) array
        Cleared binary image.  Note that the input label image is modified.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.segmentation import clear_border
    >>> labels = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0],
    ...                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    ...                    [1, 0, 0, 1, 0, 1, 0, 0, 0],
    ...                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    ...                    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> clear_border(labels)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 1, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    """
    image = labels

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

    # Re-label, in case we are dealing with a binary image
    # and to get consistent labeling
    labels = label(image, background=0) + 1
    number = np.max(labels) + 1

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
