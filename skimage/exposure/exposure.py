import warnings
import numpy as np

from skimage import img_as_float
from skimage.util.dtype import dtype_range, dtype_limits
from skimage._shared.utils import deprecated


__all__ = ['histogram', 'cumulative_distribution', 'equalize',
           'rescale_intensity', 'adjust_gamma', 'adjust_log', 'adjust_sigmoid']


DTYPE_RANGE = dtype_range.copy()
DTYPE_RANGE.update((d.__name__, limits) for d, limits in dtype_range.items())
DTYPE_RANGE.update({'uint10': (0, 2**10 - 1),
                    'uint12': (0, 2**12 - 1),
                    'uint14': (0, 2**14 - 1),
                    'bool': dtype_range[np.bool_],
                    'float': dtype_range[np.float64]})


def histogram(image, nbins=256):
    """Return histogram of image.

    Unlike `numpy.histogram`, this function returns the centers of bins and
    does not rebin integer arrays. For integer arrays, each integer value has
    its own bin, which improves speed and intensity-resolution.

    The histogram is computed on the flattened image: for color images, the
    function should be used separately on each channel to obtain a histogram
    for each color channel.

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

    Examples
    --------
    >>> from skimage import data, exposure, util
    >>> image = util.img_as_float(data.camera())
    >>> np.histogram(image, bins=2)
    (array([107432, 154712]), array([ 0. ,  0.5,  1. ]))
    >>> exposure.histogram(image, nbins=2)
    (array([107432, 154712]), array([ 0.25,  0.75]))
    """
    sh = image.shape
    if len(sh) == 3 and sh[-1] < 4:
        warnings.warn("This might be a color image. The histogram will be "
                      "computed on the flattened image. You can instead "
                      "apply this function to each color channel.")

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
    in_range : 2-tuple (float, float) or str
        Min and max *allowed* intensity values of input image. If None, the
        *allowed* min/max values are set to the *actual* min/max values in the
        input image. Intensity values outside this range are clipped.
        If string, use data limits of dtype specified by the string.
    out_range : 2-tuple (float, float) or str
        Min and max intensity values of output image. If None, use the min/max
        intensities of the image data type. See `skimage.util.dtype` for
        details. If string, use data limits of dtype specified by the string.

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
    elif in_range in DTYPE_RANGE:
        imin, imax = DTYPE_RANGE[in_range]
    else:
        imin, imax = in_range

    if out_range is None or out_range in DTYPE_RANGE:
        out_range = dtype if out_range is None else out_range
        omin, omax = DTYPE_RANGE[out_range]
        if imin >= 0:
            omin = 0
    else:
        omin, omax = out_range

    image = np.clip(image, imin, imax)

    image = (image - imin) / float(imax - imin)
    return dtype(image * (omax - omin) + omin)


def _assert_non_negative(image):

    if np.any(image < 0):
        raise ValueError('Image Correction methods work correctly only on '
                         'images with non-negative values. Use '
                         'skimage.exposure.rescale_intensity.')


def adjust_gamma(image, gamma=1, gain=1):
    """Performs Gamma Correction on the input image.

    Also known as Power Law Transform.
    This function transforms the input image pixelwise according to the
    equation ``O = I**gamma`` after scaling each pixel to the range 0 to 1.

    Parameters
    ----------
    image : ndarray
        Input image.
    gamma : float
        Non negative real number. Default value is 1.
    gain : float
        The constant multiplier. Default value is 1.

    Returns
    -------
    out : ndarray
        Gamma corrected output image.

    Notes
    -----
    For gamma greater than 1, the histogram will shift towards left and
    the output image will be darker than the input image.

    For gamma less than 1, the histogram will shift towards right and
    the output image will be brighter than the input image.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Gamma_correction

    """
    _assert_non_negative(image)
    dtype = image.dtype.type

    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number.")

    scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])

    out = ((image / scale) ** gamma) * scale * gain
    return dtype(out)


def adjust_log(image, gain=1, inv=False):
    """Performs Logarithmic correction on the input image.

    This function transforms the input image pixelwise according to the
    equation ``O = gain*log(1 + I)`` after scaling each pixel to the range 0 to 1.
    For inverse logarithmic correction, the equation is ``O = gain*(2**I - 1)``.

    Parameters
    ----------
    image : ndarray
        Input image.
    gain : float
        The constant multiplier. Default value is 1.
    inv : float
        If True, it performs inverse logarithmic correction,
        else correction will be logarithmic. Defaults to False.

    Returns
    -------
    out : ndarray
        Logarithm corrected output image.

    References
    ----------
    .. [1] http://www.ece.ucsb.edu/Faculty/Manjunath/courses/ece178W03/EnhancePart1.pdf

    """
    _assert_non_negative(image)
    dtype = image.dtype.type
    scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])

    if inv:
        out = (2 ** (image / scale) - 1) * scale * gain
        return dtype(out)

    out = np.log2(1 + image / scale) * scale * gain
    return dtype(out)


def adjust_sigmoid(image, cutoff=0.5, gain=10, inv=False):
    """Performs Sigmoid Correction on the input image.

    Also known as Contrast Adjustment.
    This function transforms the input image pixelwise according to the
    equation ``O = 1/(1 + exp*(gain*(cutoff - I)))`` after scaling each pixel
    to the range 0 to 1.

    Parameters
    ----------
    image : ndarray
        Input image.
    cutoff : float
        Cutoff of the sigmoid function that shifts the characteristic curve
        in horizontal direction. Default value is 0.5.
    gain : float
        The constant multiplier in exponential's power of sigmoid function.
        Default value is 10.
    inv : bool
        If True, returns the negative sigmoid correction. Defaults to False.

    Returns
    -------
    out : ndarray
        Sigmoid corrected output image.

    References
    ----------
    .. [1] Gustav J. Braun, "Image Lightness Rescaling Using Sigmoidal Contrast
           Enhancement Functions",
           http://www.cis.rit.edu/fairchild/PDFs/PAP07.pdf

    """
    _assert_non_negative(image)
    dtype = image.dtype.type
    scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])

    if inv:
        out = (1 - 1 / (1 + np.exp(gain * (cutoff - image/scale)))) * scale
        return dtype(out)

    out = (1 / (1 + np.exp(gain * (cutoff - image/scale)))) * scale
    return dtype(out)
