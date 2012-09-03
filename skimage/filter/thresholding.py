__all__ = ['threshold_otsu', 'threshold_adaptive']

import numpy as np
import scipy.ndimage
from skimage.exposure import histogram


def threshold_adaptive(image, block_size, method='gaussian', offset=0,
                       mode='reflect', param=None):
    """Applies an adaptive threshold to an array.

    Also known as local or dynamic thresholding where the threshold value is the
    weighted mean for the local neighborhood of a pixel subtracted by a
    constant. Alternatively the threshold can be determined dynamically by a
    a given function using the 'generic' method.

    Parameters
    ----------
    image : (N, M) ndarray
        Input image.
    block_size : int
        Uneven size of pixel neighborhood which is used to calculate the
        threshold value (e.g. 3, 5, 7, ..., 21, ...).
    method : {'generic', 'gaussian', 'mean', 'median'}, optional
        Method used to determine adaptive threshold for local neighbourhood in
        weighted mean image.

        * 'generic': use custom function (see `param` parameter)
        * 'gaussian': apply gaussian filter (see `param` parameter for custom
                      sigma value)
        * 'mean': apply arithmetic mean filter
        * 'median' apply median rank filter

        By default the 'gaussian' method is used.
    offset : float, optional
        Constant subtracted from weighted mean of neighborhood to calculate
        the local threshold value. Default offset is 0.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
        Default is 'reflect'.
    param : {int, function}, optional
        Either specify sigma for 'gaussian' method or function object for
        'generic' method. This functions takes the flat array of local
        neighbourhood as a single argument and returns the calculated threshold
        for the centre pixel.

    Returns
    -------
    threshold : (N, M) ndarray
        Thresholded binary image

    References
    ----------
    http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations
        .html?highlight=threshold#adaptivethreshold

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> binary_image1 = threshold_adaptive(image, 15, 'mean')
    >>> func = lambda arr: arr.mean()
    >>> binary_image2 = threshold_adaptive(image, 15, 'generic', param=func)
    """
    thresh_image = np.zeros(image.shape, 'double')
    if method == 'generic':
        scipy.ndimage.generic_filter(image, param, block_size,
            output=thresh_image, mode=mode)
    elif method == 'gaussian':
        if param is None:
            # automatically determine sigma which covers > 99% of distribution
            sigma = (block_size - 1) / 6.0
        else:
            sigma = param
        scipy.ndimage.gaussian_filter(image, sigma, output=thresh_image,
            mode=mode)
    elif method == 'mean':
        mask = 1. / block_size * np.ones((block_size,))
        # separation of filters to speedup convolution
        scipy.ndimage.convolve1d(image, mask, axis=0, output=thresh_image,
            mode=mode)
        scipy.ndimage.convolve1d(thresh_image, mask, axis=1,
            output=thresh_image, mode=mode)
    elif method == 'median':
        scipy.ndimage.median_filter(image, block_size, output=thresh_image,
            mode=mode)

    return image > (thresh_image - offset)


def threshold_otsu(image, nbins=256):
    """Return threshold value based on Otsu's method.

    Parameters
    ----------
    image : array
        Input image.
    nbins : int
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.

    Returns
    -------
    threshold : float
        Threshold value.

    References
    ----------
    .. [1] Wikipedia, http://en.wikipedia.org/wiki/Otsu's_Method

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_otsu(image)
    >>> binary = image > thresh
    """
    hist, bin_centers = histogram(image, nbins)
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:])**2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold
