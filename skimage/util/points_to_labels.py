import numpy as np

__all__ = ["label_points"]


def label_points(coords, image):
    """Assign unique integer labels to coordinates on an image mask

    Parameters
    ----------
    coords: ndarray
        An array of 2D coordinates with shape (n, 2)
    image: ndarray
        Image for which the mask is created

    Returns
    -------
    label_mask: ndarray
        A 2D mask of zeroes containing unique integer labels at the `coords`

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util.points_to_labels import label_points
    >>> coords = np.array([[0, 1], [2, 2]])
    >>> image = np.zeros((5, 5), dtype=np.uint32)
    >>> mask = label_points(coords, image)
    >>> mask
    array([[0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 2, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint64)
    """
    if not isinstance(coords, np.ndarray):
        raise TypeError("'coords' must be an ndarray")

    if coords.shape[-1] != 2 or coords.ndim != 2:
        raise ValueError("'coords' must be of shape (n, 2)")

    h, w = image.shape[:2]
    np_indices = tuple(np.transpose(np.round(coords).astype(np.int)))
    label_mask = np.zeros((h, w), dtype=np.uint64)
    label_mask[np_indices] = np.arange(1, coords.shape[0] + 1)
    return label_mask
