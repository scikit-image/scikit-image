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
from __future__ import division
import numbers
import numpy as np
from ..util import img_as_float, img_as_uint
from ..color.adapt_rgb import adapt_rgb, hsv_value
from ..exposure import rescale_intensity


NR_OF_GREY = 2 ** 14  # number of grayscale levels to use in CLAHE algorithm


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
    kernel_size: integer or list-like, optional
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
        Equalized image.
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
    References
    ----------
    .. [1] http://tog.acm.org/resources/GraphicsGems/
    .. [2] https://en.wikipedia.org/wiki/CLAHE#CLAHE
    """
    image = img_as_uint(image)
    image = rescale_intensity(image, out_range=(0, NR_OF_GREY - 1))

    if kernel_size is None:
        kernel_size = tuple([image.shape[dim] // 8 for dim in range(image.ndim)])
    elif isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size,) * image.ndim
    elif len(kernel_size) != image.ndim:
        ValueError('Incorrect value of `kernel_size`: {}'.format(kernel_size))

    kernel_size = [int(k) for k in kernel_size]

    image = _clahe(image, kernel_size, clip_limit * nbins, nbins)
    image = img_as_float(image)
    return rescale_intensity(image)


def _clahe(image, kernel_size, clip_limit, nbins=128):
    """Contrast Limited Adaptive Histogram Equalization.
    Parameters
    ----------
    image : (N1,...,NN) ndarray
        Input image.
    kernel_size: int or N-tuple of int
        Defines the shape of contextual regions used in the algorithm.
    clip_limit : float
        Normalized clipping limit (higher values give more contrast).
    nbins : int, optional
        Number of gray bins for histogram ("data range").
    Returns
    -------
    out : (N1,...,NN) ndarray
        Equalized image.
    The number of "effective" greylevels in the output image is set by `nbins`;
    selecting a small value (eg. 128) speeds up processing and still produce
    an output image of good quality. The output image will have the same
    minimum and maximum value as the input image. A clip limit smaller than 1
    results in standard (non-contrast limited) AHE.
    """

    if clip_limit == 1.0:
        return image  # is OK, immediately returns original image.

    ns = [int(np.ceil(image.shape[dim] / kernel_size[dim])) for dim in range(image.ndim)]

    steps = [int(np.floor(image.shape[dim] / ns[dim])) for dim in range(image.ndim)]

    bin_size = 1 + NR_OF_GREY // nbins
    lut = np.arange(NR_OF_GREY)
    lut //= bin_size

    map_array = np.zeros(tuple(ns) + (nbins,), dtype=int)

    # Calculate greylevel mappings for each contextual region

    for inds in np.ndindex(*ns):

        region = tuple([slice(inds[dim] * steps[dim], (inds[dim] + 1) * steps[dim]) for dim in range(image.ndim)])
        sub_img = image[region]

        if clip_limit > 0.0:  # Calculate actual cliplimit
            clim = int(clip_limit * sub_img.size / nbins)
            if clim < 1:
                clim = 1
        else:
            clim = NR_OF_GREY  # Large value, do not clip (AHE)

        hist = lut[sub_img.ravel()]
        hist = np.bincount(hist)
        hist = np.append(hist, np.zeros(nbins - hist.size, dtype=int))
        hist = clip_histogram(hist, clim)
        hist = map_histogram(hist, 0, NR_OF_GREY - 1, sub_img.size)
        map_array[inds] = hist

    # Interpolate greylevel mappings to get CLAHE image

    offsets = [0] * image.ndim
    lowers = [0] * image.ndim
    uppers = [0] * image.ndim
    starts = [0] * image.ndim
    prev_inds = [0] * image.ndim

    for inds in np.ndindex(*[ns[dim]+1 for dim in range(image.ndim)]):

        for dim in range(image.ndim):
            if inds[dim] != prev_inds[dim]:
                starts[dim] += offsets[dim]

        for dim in range(image.ndim):
            if dim < image.ndim - 1:
                if inds[dim] != prev_inds[dim]:
                    starts[dim+1] = 0

        prev_inds = inds[:]

        # modify edges to handle special cases
        for dim in range(image.ndim):
            if inds[dim] == 0:
                offsets[dim] = steps[dim] / 2.0
                lowers[dim] = 0
                uppers[dim] = 0
            elif inds[dim] == ns[dim]:
                offsets[dim] = steps[dim] / 2.0
                lowers[dim] = ns[dim] - 1
                uppers[dim] = ns[dim] - 1
            else:
                offsets[dim] = steps[dim]
                lowers[dim] = inds[dim] - 1
                uppers[dim] = inds[dim]

        maps = []
        for edge in np.ndindex(*([2]*image.ndim)):
            maps.append(map_array[tuple([[lowers, uppers][edge[dim]][dim] for dim in range(image.ndim)])])

        slices = [np.arange(starts[dim], starts[dim] + offsets[dim]) for dim in range(image.ndim)]

        interpolate(image, slices[::-1], maps, lut)

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

    # Second part: clip histogram and redistribute excess pixels in each bin
    bin_incr = int(n_excess / hist.size)  # average binincrement
    upper = clip_limit - bin_incr  # Bins larger than upper set to cliplimit

    hist[excess_mask] = clip_limit

    low_mask = hist < upper
    n_excess -= hist[low_mask].size * bin_incr
    hist[low_mask] += bin_incr

    mid_mask = (hist >= upper) & (hist < clip_limit)
    mid = hist[mid_mask]
    n_excess -= mid.size * clip_limit - mid.sum()
    hist[mid_mask] = clip_limit

    prev_n_excess = n_excess

    while n_excess > 0:  # Redistribute remaining excess
        index = 0
        while n_excess > 0 and index < hist.size:
            under_mask = hist < 0
            step_size = int(hist[hist < clip_limit].size / n_excess)
            step_size = max(step_size, 1)
            indices = np.arange(index, hist.size, step_size)
            under_mask[indices] = True
            under_mask = (under_mask) & (hist < clip_limit)
            hist[under_mask] += 1
            n_excess -= under_mask.sum()
            index += 1
        # bail if we have not distributed any excess
        if prev_n_excess == n_excess:
            break
        prev_n_excess = n_excess

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
    scale = ((float)(max_val - min_val)) / n_pixels
    out *= scale
    out += min_val
    out[out > max_val] = max_val
    return out.astype(int)


def interpolate(image, slices, maps, lut):
    """Find the new grayscale level for a region using bilinear interpolation.
    Parameters
    ----------
    image : ndarray
        Full image.
    slices : list of array-like
       Indices of the region.
    maps : list of ndarray
        Mappings of greylevels from histograms.
    lut : ndarray
        Maps grayscale levels in image to histogram levels.
    Returns
    -------
    out : ndarray
        Original image with the subregion replaced.
    Notes
    -----
    This function calculates the new greylevel assignments of pixels within
    a submatrix of the image. This is done by linear interpolation between
    2**image.ndim different adjacent mappings in order to eliminate boundary artifacts.
    """

    norm = np.product([slices[dim].size for dim in range(image.ndim)])  # Normalization factor

    # interpolation weight matrices
    coeffs = np.meshgrid(*tuple([np.arange(slices[dim].size) for dim in range(image.ndim)]), indexing='ij')
    coeffs = [coeff.transpose() for coeff in coeffs]

    inv_coeffs = [np.flip(coeffs[dim], axis=image.ndim - dim - 1) + 1 for dim in range(image.ndim)]

    region = tuple([slice(int(slices[dim][0]), int(slices[dim][-1] + 1)) for dim in range(image.ndim)][::-1])
    view = image[region]

    im_slice = lut[view]

    new = np.zeros_like(view, dtype=int)
    for iedge, edge in enumerate(np.ndindex(*([2]*image.ndim))):
        edge = edge[::-1]
        new += np.product([[inv_coeffs,coeffs][edge[dim]][dim] for dim in range(image.ndim)],0) * maps[iedge][im_slice]

    new = (new / norm).astype(view.dtype)
    view[::] = new
    return image