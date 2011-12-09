import numpy as np


__all__ = ['otsu_threshold', 'binarize']


def otsu_threshold(image, bins=256):
    """Return threshold value based on Otsu's method.

    Parameters
    ----------
    image : array
        Input image.
    bins : int
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.

    Returns
    -------
    threshold : numeric
        Threshold value. int or float depending on input image.

    References
    ----------
    .. [1] Wikipedia, http://en.wikipedia.org/wiki/Otsu's_Method

    """
    hist, bin_centers = histogram(image, bins)
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


_threshold_funcs = {'otsu': otsu_threshold}
def binarize(image, method='otsu'):
    """Return binary image using an automatic thresholding method.

    Parameters
    ----------
    image : array
        Input array.
    method : {'otsu'}
        Method used to calculate threshold value. Currently, only Otsu's method
        is implemented.

    Returns
    -------
    out : array
        Thresholded image.
    """
    get_threshold = _threshold_funcs[method]
    threshold = get_threshold(image)
    return image > threshold


def histogram(image, bins):
    """Return histogram of image.

    Unlike `numpy.histogram`, this function returns the centers of bins and
    does not rebin integer arrays.

    Parameters
    ----------
    image : array
        Input image.
    bins : int
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.

    Returns
    -------
    hist : array
        The values of the histogram.
    bin_centers : array
        The values at the center of the bins.
    """
    if np.issubdtype(image.dtype, np.integer):
        if np.min(image) < 0:
            msg = "Images with negative values not allowed"
            raise NotImplementedError(msg)
        hist = np.bincount(image.flat)
        bin_centers = np.arange(len(hist))

        # clip histogram to return only non-zero bins
        idx = np.nonzero(hist)[0][0]
        return hist[idx:], bin_centers[idx:]
    else:
        hist, bin_edges = np.histogram(image, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
        return hist, bin_centers

