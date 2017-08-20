"""
Identify function.

Return original image.
"""


def identity(image, **kwargs):
    """Identity function.

    Parameters
    ----------
    image : array-like
        Input image.
    kwargs : dictionary
        Dictionary of ignored keyword arguments.

    Returns
    -------
    image : ndarray
        Return original image.

    Notes
    -----
    This function is useful for automated plotting of images processed with
    different filters and comparison to the original.
    """

    return image
