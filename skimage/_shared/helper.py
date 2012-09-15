from skimage.color import rgb2gray


def ensure_gray(image, convert=True):
    """Ensure image is grayscale.

    If the image is not grayscale, it is either converted to a grayscale image
    or an exception is raised.

    Parameters
    ----------
    image : ndarray
        Input image.
    convert : bool
        Whether to automatically convert image to grayscale.

    Returns
    -------
    image : ndarray
        Greyscale image.

    """

    if image.ndim > 2:
        channels = image.shape[2]
    else:
        channels = 1
    if channels != 1:
        if convert:
            image = rgb2gray(image)
        else:
            raise ValueError('Invalid number of channels. '
                             'Grayscale image needed.')
    return image


ensure_grey = ensure_gray
