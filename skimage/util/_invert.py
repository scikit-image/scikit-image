import numpy as np
from .dtype import dtype_limits


def invert(image):
    """Invert an image.

    Substract the image to the maximum value allowed by the dtype maximum.

    Parameters
    ----------
    image : ndarray
        The input image.

    Returns
    -------
    invert : ndarray
        Inverted image.

    Examples
    --------
    >>> img = np.array([[100, 0, 200],
    ...                 [0,  50, 0],
    ...                 [30,  0, 255]], np.uint8)
    >>> invert(img)
    array([[155, 255,  55],
           [255, 205, 255],
           [225, 255,   0]], dtype=uint8)
    """
    if image.dtype == 'bool':
        return ~image
    else:
        return dtype_limits(image, clip_negative=False)[1] - image
