import numpy as np
from scipy.ndimage.filters import uniform_filter


def _guided_filter(image, guide, eta, radius):
    """ Guided filtering on two distinct single channel float images."""
    window_size = 2*radius + 1
    mean_image = uniform_filter(image, window_size)
    mean_guide = uniform_filter(guide, window_size)
    mean_image_guide = uniform_filter(guide*image, window_size)
    mean_guide_squared = uniform_filter(guide**2.0, window_size)

    a = ((mean_image_guide - mean_guide*mean_image) /
         (mean_guide_squared - mean_guide**2 + eta))  # Equation 5, [1]
    b = mean_image - a*mean_guide

    # smooth estimates in place.
    uniform_filter(a, window_size, output=a)
    uniform_filter(b, window_size, output=b)

    return a*guide + b


def _guided_filter_same(image, eta, radius):
    """ Guided filtering with a single channel image as its own guide."""
    window_size = 2*radius + 1
    mean_image = uniform_filter(image, window_size)
    mean_image_squared = uniform_filter(image**2.0, window_size)

    # use b as a temporary variable for numerator and denominator:
    b = mean_image_squared - mean_image**2
    a = b/(b + eta)
    b = mean_image*(1 - a)

    # smooth estimates in place.
    uniform_filter(a, window_size, output=a)
    uniform_filter(b, window_size, output=b)

    return a*image + b


def guided_filter(image, eta, radius, guide=None):
    """
    Guided filter: a fast edge preserving filter.

    Parameters
    ----------
    image : array-like
        Input image to filter. Can be a single or multiple channel image.
    eta : float
        Regularisation parameter, larger values correspond to smoother output.
        The effect of eta depends on the pixel intensities - for intensities in
        the range [0,1] try 0.1 as a starting point.
    radius : int
        The half edge size of the window used to compute pixel statistics for
        smoothing. The actual filter window size is 2*radius + 1.
    guide : array-like, optional
        Guide image to control the smoothing. If no guide is specified the
        image will be used as its own guide. The guide image must have the same
        number of rows and columns as the input image. If the input image has
        multiple channels the guide image must either have one channel, or the
        same number of channels as the input. If the guide has only one channel
        each channel in the input is smoothed with the same guide, otherwise
        each channel in the input is used to guide the output.

    Returns
    -------
    filtered_image : ndarray
        The filtered array, as a float.

    References
    ----------
    .. [1] "Guided Image Filtering", by He et al. appearing in ECCV 2010.
    http://research.microsoft.com/en-us/um/people/kahe/eccv10/

    Notes
    -----
    Currently uses scipy uniform_filter, so is not O(n) as implemented in the
    paper.

    Examples
    --------
    >>> a = np.eye(3)
    # If no guide is specified, the image itself is used.
    >>> guided_filter(a, 0.1, 1)
    array([[ 0.82811797,  0.10548081,  0.08231598],
           [ 0.10548081,  0.75720884,  0.10548081],
           [ 0.08231598,  0.10548081,  0.82811797]])
    # An explicit guide image can be specified.
    >>> guided_filter(a, 0.1, 1, guide=a*2)
    array([[ 0.94438564,  0.03506486,  0.02870165],
           [ 0.03506486,  0.91356598,  0.03506486],
           [ 0.02870165,  0.03506486,  0.94438564]])
    # A multichannel image can be guided with a single channel image.
    >>> a = np.tile(a[..., np.newaxis], (1, 1, 3))
    >>> b = a[:, :, 0]**2
    >>> guided_filter(a, 0.1, 1, guide=b)
    array([[[ 0.82811797,  0.82811797,  0.82811797],
        [ 0.10548081,  0.10548081,  0.10548081],
        [ 0.08231598,  0.08231598,  0.08231598]],
       [[ 0.10548081,  0.10548081,  0.10548081],
        [ 0.75720884,  0.75720884,  0.75720884],
        [ 0.10548081,  0.10548081,  0.10548081]],
       [[ 0.08231598,  0.08231598,  0.08231598],
        [ 0.10548081,  0.10548081,  0.10548081],
        [ 0.82811797,  0.82811797,  0.82811797]]])
    # Or with a multi-channel image.
    >>> b = a**2
    >>> guided_filter(a, 0.1, 1, guide=b)
    array([[[ 0.82811797,  0.82811797,  0.82811797],
            [ 0.10548081,  0.10548081,  0.10548081],
            [ 0.08231598,  0.08231598,  0.08231598]],
           [[ 0.10548081,  0.10548081,  0.10548081],
            [ 0.75720884,  0.75720884,  0.75720884],
            [ 0.10548081,  0.10548081,  0.10548081]],
           [[ 0.08231598,  0.08231598,  0.08231598],
            [ 0.10548081,  0.10548081,  0.10548081],
            [ 0.82811797,  0.82811797,  0.82811797]]])
    """
    if guide is not None:
        if image.shape[:2] != guide.shape[:2]:
            raise ValueError(
                "Image and guide must have the same first two dimensions.")
        if image.ndim == guide.ndim == 2:
            return _guided_filter(image.astype('float'),
                                  guide.astype('float'),
                                  eta, radius)
        elif image.ndim == 3 and guide.ndim == 2:
            output = np.empty(image.shape)
            for channel in range(image.shape[2]):
                output[..., channel] = _guided_filter(
                    image[..., channel].astype('float'),
                    guide.astype('float'),
                    eta, radius)
            return output
        elif image.ndim == 3 and guide.ndim == 3:
            if image.shape[2] != guide.shape[2]:
                raise ValueError(
                    "Image and guide must have the same number of channels.")
            output = np.empty(image.shape)
            for channel in range(image.shape[2]):
                output[..., channel] = _guided_filter(
                    image[..., channel].astype('float'),
                    guide[..., channel].astype('float'),
                    eta, radius)
            return output
        else:
            raise ValueError("Image and guide must have 2 or 3 dimensions")
    else:
        if image.ndim == 2:
            return _guided_filter_same(image.astype('float'),
                                       eta, radius)
        elif image.ndim == 3:
            output = np.empty(image.shape)
            for channel in range(image.shape[2]):
                output[..., channel] = _guided_filter_same(
                    image[..., channel].astype('float'),
                    eta, radius)
            return output
        else:
            raise ValueError("Image must have 2 or 3 dimensions")
