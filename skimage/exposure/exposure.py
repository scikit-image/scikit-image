import numpy as np

import skimage


__all__ = ['histogram', 'cumulative_distribution', 'equalize_hist']


def histogram(image, nbins, density=True):
    """Return histogram of image.

    Unlike `numpy.histogram`, this function returns the centers of bins and
    does not rebin integer arrays.

    Parameters
    ----------
    image : array
        Input image.
    nbins : int
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    density : {True | False}
        If True, values represent the probability density function.
        If False, values represent the number of pixels in bins.
        See numpy.histogram for details.

    Returns
    -------
    hist : array
        The values of the histogram.
    bin_centers : array
        The values at the center of the bins.
    """
    if np.issubdtype(image.dtype, np.integer):
        offset = 0
        if np.min(image) < 0:
            offset = np.min(image)
        hist = np.bincount(image.ravel() - offset)
        bin_centers = np.arange(len(hist)) + offset

        # clip histogram to start with a non-zero bin
        idx = np.nonzero(hist)[0][0]
        return hist[idx:], bin_centers[idx:]
    else:

        if np.version.version >= '1.6':
            hist, bin_edges = np.histogram(image.flat, nbins, density=density)
        else:
            hist, bin_edges = np.histogram(image.flat, nbins, normed=density)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
        return hist, bin_centers


def cumulative_distribution(image, nbins=256):
    """Return cumulative distribution function (cdf) for the given image.

    Parameters
    ----------
    image : array
        Image array.
    nbins : int
        Number of bins for image histogram.

    Returns
    -------
    img_cdf : array
        Values of cumulative distribution function.
    bin_centers : array
        Centers of bins.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Cumulative_distribution_function

    """
    hist, bin_centers = histogram(image, nbins)
    img_cdf = hist.cumsum()
    img_cdf = img_cdf / float(img_cdf[-1])
    return img_cdf, bin_centers


def equalize_hist(image, nbins=256):
    """Return image after histogram equalization.

    Parameters
    ----------
    image : array
        Image array.
    nbins : int
        Number of bins for image histogram.

    Returns
    -------
    out : float array
        Image array after histogram equalization.

    Notes
    -----
    This function is adapted from [1] with the author's permission.

    References
    ----------
    .. [1] http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    .. [2] http://en.wikipedia.org/wiki/Histogram_equalization

    """
    image = skimage.img_as_float(image)
    cdf, bin_centers = cumulative_distribution(image, nbins)
    out = np.interp(image.flat, bin_centers, cdf)
    return out.reshape(image.shape)

