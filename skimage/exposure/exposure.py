import numpy as np

from skimage import img_as_float
from skimage.util.dtype import dtype_range


__all__ = ['histogram', 'cumulative_distribution', 'equalize',
           'rescale_intensity']


def histogram(image, nbins=256):
    """Return histogram of image.

    Unlike `numpy.histogram`, this function returns the centers of bins and
    does not rebin integer arrays. For integer arrays, each integer value has
    its own bin, which improves speed and intensity-resolution.

    Parameters
    ----------
    image : array
        Input image.
    nbins : int
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.

    Returns
    -------
    hist : array
        The values of the histogram.
    bin_centers : array
        The values at the center of the bins.
    """

    # For integer types, histogramming with bincount is more efficient.
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
        hist, bin_edges = np.histogram(image.flat, nbins)
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


def equalize(image, nbins=256):
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
    This function is adapted from [1]_ with the author's permission.

    References
    ----------
    .. [1] http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    .. [2] http://en.wikipedia.org/wiki/Histogram_equalization

    """
    image = img_as_float(image)
    cdf, bin_centers = cumulative_distribution(image, nbins)
    out = np.interp(image.flat, bin_centers, cdf)
    return out.reshape(image.shape)


def rescale_intensity(image, in_range=None, out_range=None):
    """Return image after stretching or shrinking its intensity levels.

    The image intensities are uniformly rescaled such that the minimum and
    maximum values given by `in_range` match those given by `out_range`.

    Parameters
    ----------
    image : array
        Image array.
    in_range : 2-tuple (float, float)
        Min and max *allowed* intensity values of input image. If None, the
        *allowed* min/max values are set to the *actual* min/max values in the
        input image.
    out_range : 2-tuple (float, float)
        Min and max intensity values of output image. If None, use the min/max
        intensities of the image data type. See `skimage.util.dtype` for
        details.

    Returns
    -------
    out : array
        Image array after rescaling its intensity. This image is the same dtype
        as the input image.

    Examples
    --------
    By default, intensities are stretched to the limits allowed by the dtype:

    >>> image = np.array([51, 102, 153], dtype=np.uint8)
    >>> rescale_intensity(image)
    array([  0, 127, 255], dtype=uint8)

    It's easy to accidentally convert an image dtype from uint8 to float:

    >>> 1.0 * image
    array([  51.,  102.,  153.])

    Use `rescale_intensity` to rescale to the proper range for float dtypes:

    >>> image_float = 1.0 * image
    >>> rescale_intensity(image_float)
    array([ 0. ,  0.5,  1. ])

    To maintain the low contrast of the original, use the `in_range` parameter:

    >>> rescale_intensity(image_float, in_range=(0, 255))
    array([ 0.2,  0.4,  0.6])

    If the min/max value of `in_range` is more/less than the min/max image
    intensity, then the intensity levels are clipped:

    >>> rescale_intensity(image_float, in_range=(0, 102))
    array([ 0.5,  1. ,  1. ])

    If you have an image with signed integers but want to rescale the image to
    just the positive range, use the `out_range` parameter:

    >>> image = np.array([-10, 0, 10], dtype=np.int8)
    >>> rescale_intensity(image, out_range=(0, 127))
    array([  0,  63, 127], dtype=int8)

    """
    dtype = image.dtype.type

    if in_range is None:
        imin = np.min(image)
        imax = np.max(image)
    else:
        imin, imax = in_range

    if out_range is None:
        omin, omax = dtype_range[dtype]
        if imin >= 0:
            omin = 0
    else:
        omin, omax = out_range

    image = np.clip(image, imin, imax)

    image = (image - imin) / float(imax - imin)
    return dtype(image * (omax - omin) + omin)
