import numpy as np

__all__ = ["label_points"]


def label_points(coords, image):
    """Assign unique integer labels to coordinates on an image mask

    Parameters
    __________
    coords: ndarray
        An array of 2D coordinates with shape (number of points, 2)
    image: ndarray
        Image for which the mask is created

    Returns
    _______
    label_mask: ndarray
        A 2D mask of zeroes containing unique integer labels at the `coords`
    """
    h, w = image.shape[:2]
    np_indices = tuple(np.transpose(np.round(coords).astype(np.int)))
    label_mask = np.zeros((h, w), dtype=np.uint32)
    label_mask[np_indices] = np.arange(1, coords.shape[0] + 1)
    return label_mask
