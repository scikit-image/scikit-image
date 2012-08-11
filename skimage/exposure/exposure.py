import numpy as np

import skimage
from skimage.util.dtype import dtype_range
import skimage.color as color
from skimage.util.dtype import convert

from _adapthist import _adapthist

__all__ = ['histogram', 'cumulative_distribution', 'equalize',
           'rescale_intensity', 'adapthist']


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
    image = skimage.img_as_float(image)
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


def adapthist(image, nx=8, ny=8, clip_limit=0.01, nbins=256, out_range='full'):
    '''Contrast Limited Adaptive Histogram Equalization

    Parameters
    ----------
    image : array-like
        original image
    nx : int, optional
        Tile regions in the X direction (2, 16)
    ny : int, optional
        Tile regions in the Y direction (2, 16)
    clip_limit : float: optional
        Normalized cliplimit (higher values give more contrast)
    nbins : int, optional
        Greybins for histogram ("dynamic range")
    out_range : str, optional
        Range of the output image data.
           - 'original' - Use original image limits
           - 'full' - Use full range of image data type

    Returns
    -------
    out : np.ndarray
        equalized image - may be a different shape than the original

    Notes
    -----
    * The underlying algorithm relies on an image whose rows and columns are even multiples of
    the number of tiles, so the extra rows and columns are left at their original values, thus
    preserving the input image shape.
    * For grayscale images, CLAHE is performed on one channel, and a grayscale is returned
    * For color images, the following steps are performed:
       - The image is converted to LAB color space
       - The CLAHE algorithm is run on the L channel
       - The image is converted back to RGB space and returned
    * For RGBA images, the original alpha channel is removed.

    References
    ----------
    .. [1] http://tog.acm.org/resources/GraphicsGems/
    .. [2] https://en.wikipedia.org/wiki/CLAHE#CLAHE
    '''
    in_type = image.dtype.type
    if out_range == 'full':
        out_range = None
    else:
        out_range = (image.min(), image.max())
    # must be converted to 12 bit for CLAHE
    int_image = skimage.img_as_uint(image)
    MAX_VAL = 2 ** 12 - 1
    int_image = rescale_intensity(int_image, out_range=(0, MAX_VAL))
    # handle color images - CLAHE accepts scalar images only
    args = [int_image.copy(), 0, MAX_VAL, nx, ny, nbins, clip_limit]
    if image.ndim == 3:
        # check for grayscale
        if (np.allclose(image[:, :, 0], image[:, :, 1]) and
            np.allclose(image[:, :, 2], image[:, :, 3])):
            args[0] = image[:, :, 0]
            out = _adapthist(*args)
            image = int_image[:, :, :3]
            for channel in range(3):
                image[:out.shape[0], :out.shape[1], channel] = out
        # for color images, convert to LAB space for processing
        else:
            lab_img = color.rgb2lab(skimage.img_as_float(image))
            L_chan = lab_img[:, :, 0]
            L_chan /= np.max(np.abs(L_chan))
            L_chan = skimage.img_as_uint(L_chan)
            args[0] = rescale_intensity(L_chan, out_range=(0, MAX_VAL))
            new_L = _adapthist(*args).astype(float)
            new_L = rescale_intensity(new_L, out_range=(0, 100))
            lab_img[:new_L.shape[0], :new_L.shape[1], 0] = new_L
            image = color.lab2rgb(lab_img)
            image = rescale_intensity(image, out_range=(0, 1))
    else:
        out = _adapthist(*args)
        image = int_image
        image[:out.shape[0], :out.shape[1]] = out
    # restore to desired output type and output limits
    image = rescale_intensity(image)
    image = convert(image, in_type)
    image = rescale_intensity(image, out_range=out_range)
    return image

if __name__ == '__main__':
    from skimage import data
    import matplotlib.pyplot as plt
    img = skimage.img_as_uint(data.lena())
    adapted = adapthist(img, nx=10, ny=9, clip_limit=0.01,
                            nbins=128, out_range='original')
    plt.imshow(img)
    plt.figure(); plt.imshow(skimage.img_as_ubyte(adapted))
    plt.figure(); plt.imshow(color.lab2rgb(color.rgb2lab(img)))
    plt.show()
    print 'Done'