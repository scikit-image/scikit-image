import numpy as np

from skimage.exposure import histogram
from ._thresholding import _threshold_adaptive


__all__ = ['threshold_otsu', 'threshold_adaptive']


def threshold_adaptive(image, block_size, method='gaussian', offset=0,
                       mode='reflect', param=None):
    """Applies an adaptive threshold to an array.

    Also known as local or dynamic thresholding where the threshold value is the
    weighted mean for the local neighborhood of a pixel subtracted by a
    constant. Alternatively the threshold can be determined dynamically by a
    a given function using the 'generic' method.

    Parameters
    ----------
    image : NxM ndarray
        Input image.
    block_size : int
        uneven size of pixel neighborhood which is used to calculate the
        threshold value (e.g. 3, 5, 7, ..., 21, ...)
    method : {'generic', 'gaussian', 'mean', 'median'}, optional
        method used to determine adaptive threshold. By default the 'gaussian'
        method is used.
    offset : float, optional
        constant subtracted from weighted mean of neighborhood to calculate
        the local threshold value. Default offset is 0.
    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'. Default is 'reflect'
    param : {int, function}, optional
        either specify sigma for 'gaussian' method or function object for
        'generic' method.

    Returns
    -------
    threshold : NxM ndarray
        thresholded binary image

    References
    ----------
    http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations
        .html?highlight=threshold#adaptivethreshold
    """
    # not using img_as_float because offset parameter wouldn't work
    image = image.astype('double')
    return _threshold_adaptive(image, block_size, method, offset, mode, param)

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
