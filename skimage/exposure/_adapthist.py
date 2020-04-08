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
    image : (M, N[, C]) ndarray
        Input image.
    kernel_size: integer or array_like, optional
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
    out : (M, N[, C]) ndarray
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

    References
    ----------
    .. [1] http://tog.acm.org/resources/GraphicsGems/
    .. [2] https://en.wikipedia.org/wiki/CLAHE#CLAHE
    """

    if clip_limit == 1.0:
        return img_as_float(image)  # convert to float for consistency

    image = img_as_uint(image)
    image = rescale_intensity(image, out_range=(0, NR_OF_GRAY - 1))

    if kernel_size is None:
        kernel_size = (image.shape[0] // 8, image.shape[1] // 8)
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
    image : (M, N) ndarray
        Input image.
    kernel_size: 2-tuple of int
        Defines the shape of contextual regions used in the algorithm.
    clip_limit : float
        Normalized clipping limit (higher values give more contrast).
    nbins : int
        Number of gray bins for histogram ("data range").

    Returns
    -------
    out : (M, N) ndarray
        Equalized image.

    The number of "effective" graylevels in the output image is set by `nbins`;
    selecting a small value (eg. 128) speeds up processing and still produce
    an output image of good quality. The output image will have the same
    minimum and maximum value as the input image. A clip limit smaller than 1
    results in standard (non-contrast limited) AHE.
    """
    row_step = int(image.shape[0] // np.ceil(image.shape[0] / kernel_size[0]))
    col_step = int(image.shape[1] // np.ceil(image.shape[1] / kernel_size[1]))

    nr = image.shape[0] // row_step
    nc = image.shape[1] // col_step

    bin_size = 1 + NR_OF_GRAY // nbins
    lut = np.arange(NR_OF_GRAY)
    lut //= bin_size

    map_array = np.zeros((nr, nc, nbins), dtype=int)

    # Calculate graylevel mappings for each contextual region
    i0 = 0
    for r in range(nr):
        i1 = i0 + row_step
        j0 = 0
        for c in range(nc):
            j1 = j0 + col_step
            sub_img = image[i0:i1, j0:j1]

            if clip_limit > 0.0:  # Calculate actual cliplimit
                clim = max(int(clip_limit * sub_img.size), 1)
            else:
                clim = NR_OF_GRAY  # Large value, do not clip (AHE)

            hist = np.bincount(lut[sub_img.ravel()], minlength=nbins)
            hist = clip_histogram(hist, clim)
            hist = map_histogram(hist, 0, NR_OF_GRAY - 1, sub_img.size)
            map_array[r, c] = hist
            j0 = j1
        i0 = i1

    # Interpolate graylevel mappings to get CLAHE image
    rstart = 0
    for r in range(nr + 1):
        cstart = 0
        rU = max(0, r - 1)
        rB = min(nr - 1, r)
        if r in [0, nr]:  # special case: top and bottom rows
            r_offset = row_step // 2
        else:  # default values
            r_offset = row_step

        rslice = slice(rstart, rstart + r_offset)

        for c in range(nc + 1):
            cL = max(0, c - 1)
            cR = min(nc - 1, c)
            if c in [0, nc]:  # special case: left and right columns
                c_offset = col_step // 2
            else:  # default values
                c_offset = col_step

            cslice = slice(cstart, cstart + c_offset)

            mapLU = map_array[rU, cL]
            mapRU = map_array[rU, cR]
            mapLB = map_array[rB, cL]
            mapRB = map_array[rB, cR]

            interpolate(image, cslice, rslice,
                        mapLU, mapRU, mapLB, mapRB, lut)

            cstart += c_offset  # set pointer on next matrix */

        rstart += r_offset

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


def interpolate(image, xslice, yslice,
                mapLU, mapRU, mapLB, mapRB, lut):
    """Find the new grayscale level for a region using bilinear interpolation.

    Parameters
    ----------
    image : ndarray
        Full image.
    xslice, yslice : slice
       Slices of the region.
    map* : ndarray
        Mappings of graylevels from histograms.
    lut : ndarray
        Maps grayscale levels in image to histogram levels.

    Returns
    -------
    out : ndarray
        Original image with the subregion replaced.

    Notes
    -----
    This function calculates the new graylevel assignments of pixels within
    a submatrix of the image. This is done by a bilinear interpolation between
    four different mappings in order to eliminate boundary artifacts.
    """
    view = image[yslice, xslice]
    y_size, x_size = view.shape
    # interpolation weight matrices
    x_coef, y_coef = np.meshgrid(np.arange(x_size), np.arange(y_size))
    x_inv_coef, y_inv_coef = x_coef[:, ::-1] + 1, y_coef[::-1] + 1

    im_slice = lut[view]
    new = (y_inv_coef * (x_inv_coef * mapLU[im_slice]
                         + x_coef * mapRU[im_slice])
           + y_coef * (x_inv_coef * mapLB[im_slice]
                       + x_coef * mapRB[im_slice]))
    view[:, :] = new / (x_size * y_size)
    return image
