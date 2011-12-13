import numpy as np


__all__ = ['cumulative_distribution', 'equalize_hist']


def cumulative_distribution(img, nbins=256):
    """Return cumulative distribution function (cdf) for the given image.

    Parameters
    ----------
    img : array
        Image array.
    nbins : int
        Number of bins for image histogram.

    Returns
    -------
    img_cdf : array
        Values of cumulative distribution function.
    bin_edges : array
        Bin edges for cdf. Length is ``len(img_cdf) + 1``.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Cumulative_distribution_function

    """
    hist, bin_edges = np.histogram(img.flat, nbins, density=True)
    img_cdf = hist.cumsum()
    return img_cdf, bin_edges


def equalize_hist(img, nbins=256, max_intensity=255):
    """Return image after histogram equalization.

    Parameters
    ----------
    img : array
        Image array.
    nbins : int
        Number of bins for image histogram.
    max_intensity : int
        Maximum intensity of the returned image.

    Returns
    -------
    out : array
        Image array after histogram equalization.

    Notes
    -----
    This function is adapted from [1] with the author's permission.

    References
    ----------
    .. [1] http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    .. [2] http://en.wikipedia.org/wiki/Histogram_equalization

    """
    cdf, bin_edges = cumulative_distribution(img, nbins)
    cdf = max_intensity * cdf / cdf[-1]
    out = np.interp(img.flat, bin_edges[:-1], cdf)
    return out.reshape(img.shape)

