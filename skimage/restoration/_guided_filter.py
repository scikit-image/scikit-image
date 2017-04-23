import numpy as np
from scipy.ndimage.filters import uniform_filter


def _guided_filter(image, guide, eta, radius):
    """Guided filtering on two distinct single channel float images."""
    window_size = 2 * radius + 1
    mean_image = uniform_filter(image, window_size)
    mean_guide = uniform_filter(guide, window_size)
    mean_image_guide = uniform_filter(guide * image, window_size)
    mean_guide_squared = uniform_filter(guide**2.0, window_size)
    a = ((mean_image_guide - mean_guide * mean_image) /
         (mean_guide_squared - mean_guide**2 + eta))
    b = mean_image - a * mean_guide

    # Smooth estimates in place
    uniform_filter(a, window_size, output=a)
    uniform_filter(b, window_size, output=b)
    return a * guide + b


def _guided_filter_same(image, eta, radius):
    """Guided filtering with a single channel image as its own guide."""
    window_size = 2 * radius + 1
    mean_image = uniform_filter(image, window_size)
    mean_image_squared = uniform_filter(image**2.0, window_size)

    # Use b as a temporary variable for numerator and denominator
    b = mean_image_squared - mean_image**2
    a = b / (b + eta)
    b = mean_image * (1 - a)

    # Smooth estimates in place
    uniform_filter(a, window_size, output=a)
    uniform_filter(b, window_size, output=b)
    return a * image + b


def guided_filter(image, eta, radius, guide=None):
    """
    Guided filter: a fast edge preserving filter.

    Parameters
    ----------
    image : ndarray, shape (M, N[, C])
        Input image to filter. Can be a single or multiple channel image.
    eta : float
        Regularization parameter, larger values correspond to smoother output.
        The effect of eta depends on the pixel intensities - for intensities in
        the range [0,1] try 0.1 as a starting point.
    radius : int
        The half edge size of the window used to compute pixel statistics for
        smoothing. The actual filter window size is 2*radius + 1.
    guide : ndarray, shape (M, N[, C]), optional
        Guide image to control the smoothing. If None, the image will be used
        as its own guide. The guide image must have the same number of rows and
        columns as the input image. If the input image has multiple channels
        the guide image must either have one channel, or the same number of
        channels as the input. If the guide has only one channel each channel
        in the input is smoothed with the same guide, otherwise each channel in
        the input is used to guide the output.

    Returns
    -------
    out : ndarray of floats
        The filtered image.

    References
    ----------
    .. [1] Guided Image Filtering, Kaiming He, Jian Sun, and Xiaoou Tang,
           http://kaiminghe.com/eccv10/index.html, 2010,
           DOI: 10.1007/978-3-642-15549-9_1

    Notes
    -----
    Currently uses scipy uniform_filter, so is not O(n) as implemented in the
    paper.

    Examples
    --------
    >>> a = np.eye(3)
    >>> # If no guide is specified, the image itself is used.
    >>> filtered = guided_filter(a, 0.1, 1)
    >>> # An explicit guide image can be specified.
    >>> filtered = guided_filter(a, 0.1, 1, guide=a*2)
    >>> # A multichannel image can be guided with a single channel image.
    >>> a = np.tile(a[..., np.newaxis], (1, 1, 3))
    >>> b = a[:, :, 0]**2
    >>> filtered = guided_filter(a, 0.1, 1, guide=b)
    >>> # Or with a multi-channel image.
    >>> b = a**2
    >>> filtered = guided_filter(a, 0.1, 1, guide=b)
    """
    image = image.astype('float')
    out = np.empty(image.shape)

    if guide is None:
        if image.ndim == 2:
            out = _guided_filter_same(image, eta, radius)
        elif image.ndim == 3:
            for channel in range(image.shape[2]):
                out[..., channel] = _guided_filter_same(
                    image[..., channel], eta, radius)
        else:
            raise ValueError("Image must have 2 or 3 dimensions")
    else:
        guide = guide.astype('float')

        if image.shape[:2] != guide.shape[:2]:
            raise ValueError(
                "Image and guide must have the same first two dimensions.")
        if image.ndim == guide.ndim == 2:
            out = _guided_filter(image, guide, eta, radius)
        elif image.ndim == 3 and guide.ndim == 2:
            for channel in range(image.shape[2]):
                out[..., channel] = _guided_filter(
                    image[..., channel], guide, eta, radius)
        elif image.ndim == 3 and guide.ndim == 3:
            if image.shape[2] != guide.shape[2]:
                raise ValueError(
                    "Image and guide must have the same number of channels.")
            for channel in range(image.shape[2]):
                out[..., channel] = _guided_filter(
                    image[..., channel], guide[..., channel], eta, radius)
        else:
            raise ValueError("Image and guide must have 2 or 3 dimensions")

    return out
