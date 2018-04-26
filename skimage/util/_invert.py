import numpy as np
from .dtype import dtype_limits


def invert(image, signed_float=False):
    """Invert an image.

    Image inversion. If the image is of type unsignedinteger, the image
    is subtracted from the maximum value allowed by the dtype maximum.
    If the image is of type signedinteger, the image is subtracted from -1.
    If the image is of float type and signed_float is true, the result is
    -image. If the image is of float type and signed_float is false (default)
    the result is 1.0 - image.

    Parameters
    ----------
    image : ndarray
        The input image.
    signed_float : bool
        If True and the image is of type float, the range is assumed to
        be [-1, 1]. If False and the image is of type float, the range is
        assumed to be [0, 1]. Default value is False (i.e. the assumed range is
        [0, 1].

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
        return(~image)
    elif np.issubdtype(image.dtype, np.unsignedinteger):
        return(dtype_limits(image, clip_negative=False)[1] - image)
    elif np.issubdtype(image.dtype, np.signedinteger):
        return(-1 - image)
    else:
        if signed_float:
            return(-image)
        else:
            return(1 - image)
