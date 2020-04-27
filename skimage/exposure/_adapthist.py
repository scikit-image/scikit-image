"""
Adapted code from "Contrast Limited Adaptive Histogram Equalization" by Karel
Zuiderveld <karel@cv.ruu.nl>, Graphics Gems IV, Academic Press, 1994.

http://tog.acm.org/resources/GraphicsGems/

The Graphics Gems code is copyright-protected.  In other words, you cannot
claim the text of the code as your own and resell it. Using the code is
permitted in any program, product, or library, non-commercial or commercial.
Giving credit is not required, though is a nice gesture.  The code comes as-is,
and if there are any flaws or problems with any Gems code, nobody involved with
Gems - authors, editors, publishers, or webmasters - are to be held
responsible.  Basically, don't be a jerk, and remember that anything free
comes with no guarantee.
"""
import numbers
import numpy as np
from ..util import img_as_float, img_as_uint
from ..color.adapt_rgb import adapt_rgb, hsv_value
from ..exposure import rescale_intensity


NR_OF_GRAY = 2 ** 14  # number of grayscale levels to use in CLAHE algorithm


@adapt_rgb(hsv_value)
def equalize_adapthist(image, kernel_size=None,
                       clip_limit=0.01, nbins=256):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE).

    An algorithm for local contrast enhancement, that uses histograms computed
    over different tile regions of the image. Local details can therefore be
    enhanced even in regions that are darker or lighter than most of the image.

    Parameters
    ----------
    image : (N1, ...,NN[, C]) ndarray
        Input image.
    kernel_size: int or array_like, optional
        Defines the shape of contextual regions used in the algorithm. If
        iterable is passed, it must have the same number of elements as
        ``image.ndim`` (without color channel). If integer, it is broadcasted
        to each `image` dimension. By default, ``kernel_size`` is 1/8 of
        ``image`` height by 1/8 of its width.
    clip_limit : float, optional
        Clipping limit, normalized between 0 and 1 (higher values give more
        contrast).
    nbins : int, optional
        Number of gray bins for histogram ("data range").

    Returns
    -------
    out : (N1, ...,NN[, C]) ndarray
        Equalized image with float64 dtype.

    See Also
    --------
    equalize_hist, rescale_intensity

    Notes
    -----
    * For color images, the following steps are performed:
       - The image is converted to HSV color space
       - The CLAHE algorithm is run on the V (Value) channel
       - The image is converted back to RGB space and returned
    * For RGBA images, the original alpha channel is removed.

    .. versionchanged:: 0.17
        The values returned by this function are slightly shifted upwards
        because of an internal change in rounding behavior.

    References
    ----------
    .. [1] http://tog.acm.org/resources/GraphicsGems/
    .. [2] https://en.wikipedia.org/wiki/CLAHE#CLAHE
    """

    if clip_limit == 1.0:
        return img_as_float(image)  # convert to float for consistency

    image = img_as_uint(image)
    image = np.round(
        rescale_intensity(image, out_range=(0, NR_OF_GRAY - 1))
    ).astype(np.uint16)

    if kernel_size is None:
        kernel_size = tuple([image.shape[dim] // 8
                             for dim in range(image.ndim)])
    elif isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size,) * image.ndim
    elif len(kernel_size) != image.ndim:
        ValueError('Incorrect value of `kernel_size`: {}'.format(kernel_size))

    kernel_size = [int(k) for k in kernel_size]

    image = _clahe(image, kernel_size, clip_limit, nbins)
    image = img_as_float(image)
    return rescale_intensity(image)


def _clahe(image, kernel_size, clip_limit, nbins):
    """Contrast Limited Adaptive Histogram Equalization.

    Parameters
    ----------
    image : (N1,...,NN) ndarray
        Input image.
    kernel_size: int or N-tuple of int
        Defines the shape of contextual regions used in the algorithm.
    clip_limit : float
        Normalized clipping limit (higher values give more contrast).
    nbins : int
        Number of gray bins for histogram ("data range").

    Returns
    -------
    out : (N1,...,NN) ndarray
        Equalized image.

    The number of "effective" graylevels in the output image is set by `nbins`;
    selecting a small value (eg. 128) speeds up processing and still produce
    an output image of good quality. The output image will have the same
    minimum and maximum value as the input image. A clip limit smaller than 1
    results in standard (non-contrast limited) AHE.
    """
    # pad the image such that the shape in each dimension
    # is a multiple of the relevant kernel_size
    pad_end_per_dim = [(k - s % k) % k
                       for k, s in zip(kernel_size, image.shape)]

    image = np.pad(image, [[0, p] for p in pad_end_per_dim],
                   mode='reflect')

    ns = [int(np.ceil(s / k)) for s, k in zip(image.shape, kernel_size)]

    steps = [int(np.floor(s / n)) for s, n in zip(image.shape, ns)]

    bin_size = 1 + NR_OF_GRAY // nbins
    lut = np.arange(NR_OF_GRAY)
    lut //= bin_size

    map_array = np.zeros(tuple(ns) + (nbins,), dtype=int)

    # calculate graylevel mappings for each contextual region
    for inds in np.ndindex(*ns):

        region = tuple([slice(i * s, (i + 1) * s)
                        for i, s in zip(inds, steps)])

        sub_img = image[region]

        if clip_limit > 0.0:  # Calculate actual clip limit
            clim = int(clip_limit * sub_img.size)
            if clim < 1:
                clim = 1
        else:
            clim = NR_OF_GRAY  # Large value, do not clip (AHE)

        hist = lut[sub_img.ravel()]
        hist = np.bincount(hist, minlength=nbins)
        hist = np.append(hist, np.zeros(nbins - hist.size, dtype=int))
        hist = clip_histogram(hist, clim)
        hist = map_histogram(hist, 0, NR_OF_GRAY - 1, sub_img.size)
        map_array[inds] = hist

    # Perform multilinear interpolation of graylevel mappings
    # using the convention described here:
    # https://en.wikipedia.org/w/index.php?title=Adaptive_histogram_
    # equalization&oldid=936814673#Efficient_computation_by_interpolation

    # determine num of blocks to be processed separately in each dim:
    # generally n+1, n if last block only processes padded pixels
    ns_proc = [n + 1 if (np.ceil(s / 2.) + (n - 1) * s) <= (sh - pe)
               else n for sh, pe, s, n
               in zip(image.shape, pad_end_per_dim, steps, ns)]

    for inds in np.ndindex(*ns_proc):

        # define slices for each dim
        starts = [int(i > 0) * np.ceil(s / 2.) + np.max([0, i - 1]) * s
                  for i, s in zip(inds, steps)]

        offsets = [np.ceil(s / 2.) if not i else s
                   for i, s, n in zip(inds, steps, ns)]

        slices = [slice(int(st), int(np.min([st + o, sh])))
                  for st, o, sh in zip(starts, offsets, image.shape)]

        # define neighboring contextual regions
        lowers = [np.max([0, i - 1]) for i in inds]
        uppers = [np.min([n - 1, i]) for i, n in zip(inds, ns)]

        maps = [map_array[tuple([[lowers, uppers][e][dim]
                                 for dim, e in enumerate(edge)])]
                for edge in np.ndindex(*([2] * image.ndim))]

        interpolate(image, slices, maps, lut)

    # undo padding
    unpad_slices = tuple([slice(0, s - p)
                          for s, p in zip(image.shape, pad_end_per_dim)])
    image = image[unpad_slices]

    return image


def clip_histogram(hist, clip_limit):
    """Perform clipping of the histogram and redistribution of bins.

    The histogram is clipped and the number of excess pixels is counted.
    Afterwards the excess pixels are equally redistributed across the
    whole histogram (providing the bin count is smaller than the cliplimit).

    Parameters
    ----------
    hist : ndarray
        Histogram array.
    clip_limit : int
        Maximum allowed bin count.

    Returns
    -------
    hist : ndarray
        Clipped histogram.
    """
    # calculate total number of excess pixels
    excess_mask = hist > clip_limit
    excess = hist[excess_mask]
    n_excess = excess.sum() - excess.size * clip_limit
    hist[excess_mask] = clip_limit

    # Second part: clip histogram and redistribute excess pixels in each bin
    bin_incr = n_excess // hist.size  # average binincrement
    upper = clip_limit - bin_incr  # Bins larger than upper set to cliplimit

    low_mask = hist < upper
    n_excess -= hist[low_mask].size * bin_incr
    hist[low_mask] += bin_incr

    mid_mask = np.logical_and(hist >= upper, hist < clip_limit)
    mid = hist[mid_mask]
    n_excess += mid.sum() - mid.size * clip_limit
    hist[mid_mask] = clip_limit

    while n_excess > 0:  # Redistribute remaining excess
        prev_n_excess = n_excess
        for index in range(hist.size):
            under_mask = hist < clip_limit
            step_size = max(1, np.count_nonzero(under_mask) // n_excess)
            under_mask = under_mask[index::step_size]
            hist[index::step_size][under_mask] += 1
            n_excess -= np.count_nonzero(under_mask)
            if n_excess <= 0:
                break
        if prev_n_excess == n_excess:
            break

    return hist


def map_histogram(hist, min_val, max_val, n_pixels):
    """Calculate the equalized lookup table (mapping).

    It does so by cumulating the input histogram.

    Parameters
    ----------
    hist : ndarray
        Clipped histogram.
    min_val : int
        Minimum value for mapping.
    max_val : int
        Maximum value for mapping.
    n_pixels : int
        Number of pixels in the region.

    Returns
    -------
    out : ndarray
       Mapped intensity LUT.
    """
    out = np.cumsum(hist).astype(float)
    out *= (max_val - min_val) / n_pixels
    out += min_val
    np.clip(out, a_min=None, a_max=max_val, out=out)
    return out.astype(int)


def interpolate(image, slices, maps, lut):
    """Find the new grayscale level for a region
    using multilinear interpolation.

    Parameters
    ----------
    image : ndarray
        Full image.
    slices : list of slices
       Indices of the region.
    maps : list of ndarray
        Mappings of graylevels from histograms.
    lut : ndarray
        Maps grayscale levels in image to histogram levels.

    Returns
    -------
    out : ndarray
        Original image with the subregion replaced.

    Notes
    -----
    This function calculates the new graylevel assignments of pixels
    within a submatrix of the image. This is done by multilinear
    interpolation between 2^image.ndim different adjacent mappings
    in order to eliminate boundary artifacts.
    """
    region = tuple([s for s in slices])
    view = image[region]

    # interpolation weight matrices
    coeffs = np.meshgrid(*tuple([np.arange(s) for s in view.shape[::-1]]),
                         indexing='ij')
    coeffs = [np.transpose(c) for c in coeffs]

    inv_coeffs = [np.flip(c, axis=image.ndim - dim - 1) + 1
                  for dim, c in enumerate(coeffs)]

    im_slice = lut[view]

    result = np.zeros_like(view, dtype=int)
    for iedge, edge in enumerate(np.ndindex(*([2] * image.ndim))):
        result += (np.product([[inv_coeffs, coeffs][e][dim]
                               for dim, e in enumerate(edge[::-1])], 0)
                   * maps[iedge][im_slice])

    # normalize
    result = result / np.product(view.shape)

    view[::] = result.astype(view.dtype)
    return image
