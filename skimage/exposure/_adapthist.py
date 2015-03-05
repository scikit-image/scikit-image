"""
Adapted code from "Contrast Limited Adaptive Histogram Equalization" by Karel
Zuiderveld <karel@cv.ruu.nl>, Graphics Gems IV, Academic Press, 1994.

http://tog.acm.org/resources/GraphicsGems/gems.html#gemsvi

The Graphics Gems code is copyright-protected.  In other words, you cannot
claim the text of the code as your own and resell it. Using the code is
permitted in any program, product, or library, non-commercial or commercial.
Giving credit is not required, though is a nice gesture.  The code comes as-is,
and if there are any flaws or problems with any Gems code, nobody involved with
Gems - authors, editors, publishers, or webmasters - are to be held
responsible.  Basically, don't be a jerk, and remember that anything free
comes with no guarantee.
"""
import numpy as np
from .. import img_as_float, img_as_uint
from ..color.adapt_rgb import adapt_rgb, hsv_value
from ..exposure import rescale_intensity
from ..util import view_as_blocks, pad


MAX_REG_X = 16  # max. # contextual regions in x-direction */
MAX_REG_Y = 16  # max. # contextual regions in y-direction */
NR_OF_GREY = 2 ** 14  # number of grayscale levels to use in CLAHE algorithm


@adapt_rgb(hsv_value)
def equalize_adapthist(image, ntiles_x=8, ntiles_y=8, clip_limit=0.01,
                       nbins=256):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE).

    An algorithm for local contrast enhancement, that uses histograms computed
    over different tile regions of the image. Local details can therefore be
    enhanced even in regions that are darker or lighter than most of the image.

    Parameters
    ----------
    image : array-like
        Input image.
    ntiles_x : int, optional
        Number of tile regions in the X direction.  Ranges between 1 and 16.
    ntiles_y : int, optional
        Number of tile regions in the Y direction.  Ranges between 1 and 16.
    clip_limit : float: optional
        Clipping limit, normalized between 0 and 1 (higher values give more
        contrast).
    nbins : int, optional
        Number of gray bins for histogram ("dynamic range").

    Returns
    -------
    out : ndarray
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
    * The CLAHE algorithm relies on image blocks of equal size.  This may
      result in extra border pixels that would not be handled.  In that case,
      we pad the image with a repeat of the border pixels, apply the
      algorithm, and then trim the image to original size.

    References
    ----------
    .. [1] http://tog.acm.org/resources/GraphicsGems/gems.html#gemsvi
    .. [2] https://en.wikipedia.org/wiki/CLAHE#CLAHE
    """
    image = img_as_uint(image)
    image = rescale_intensity(image, out_range=(0, NR_OF_GREY - 1))
    out = _clahe(image, ntiles_x, ntiles_y, clip_limit * nbins, nbins)
    image[:out.shape[0], :out.shape[1]] = out
    image = img_as_float(image)
    return rescale_intensity(image)


def _clahe(image, ntiles_x, ntiles_y, clip_limit, nbins=128):
    """Contrast Limited Adaptive Histogram Equalization.

    Parameters
    ----------
    image : array-like
        Input image.
    ntiles_x : int, optional
        Number of tile regions in the X direction.  Ranges between 2 and 16.
    ntiles_y : int, optional
        Number of tile regions in the Y direction.  Ranges between 2 and 16.
    clip_limit : float, optional
        Normalized clipping limit (higher values give more contrast).
    nbins : int, optional
        Number of gray bins for histogram ("dynamic range").

    Returns
    -------
    out : ndarray
        Equalized image.

    The number of "effective" greylevels in the output image is set by `nbins`;
    selecting a small value (eg. 128) speeds up processing and still produce
    an output image of good quality. The output image will have the same
    minimum and maximum value as the input image. A clip limit smaller than 1
    results in standard (non-contrast limited) AHE.
    """
    ntiles_x = min(ntiles_x, MAX_REG_X)
    ntiles_y = min(ntiles_y, MAX_REG_Y)

    if clip_limit == 1.0:
        return image  # is OK, immediately returns original image.

    h_inner = image.shape[0] - image.shape[0] % ntiles_y
    w_inner = image.shape[1] - image.shape[1] % ntiles_x

    # make the tile size divisible by 2
    h_inner -= h_inner % (2 * ntiles_y)
    w_inner -= w_inner % (2 * ntiles_x)

    orig_shape = image.shape
    width = w_inner // ntiles_x  # Actual size of contextual regions
    height = h_inner // ntiles_y

    if h_inner != image.shape[0]:
        ntiles_y += 1
    if w_inner != image.shape[1]:
        ntiles_x += 1
    if h_inner != image.shape[1] or w_inner != image.shape[0]:
        h_pad = height * ntiles_y - image.shape[0]
        w_pad = width * ntiles_x - image.shape[1]
        image = pad(image, ((0, h_pad), (0, w_pad)), mode='reflect')
        h_inner, w_inner = image.shape

    bin_size = 1 + NR_OF_GREY // nbins
    lut = np.arange(NR_OF_GREY)
    lut //= bin_size
    img_blocks = view_as_blocks(image, (height, width))

    map_array = np.zeros((ntiles_y, ntiles_x, nbins), dtype=int)
    n_pixels = width * height

    if clip_limit > 0.0:  # Calculate actual cliplimit
        clip_limit = int(clip_limit * (width * height) / nbins)
        if clip_limit < 1:
            clip_limit = 1
    else:
        clip_limit = NR_OF_GREY  # Large value, do not clip (AHE)

    # Calculate greylevel mappings for each contextual region
    for y in range(ntiles_y):
        for x in range(ntiles_x):
            sub_img = img_blocks[y, x]
            hist = lut[sub_img.ravel()]
            hist = np.bincount(hist)
            hist = np.append(hist, np.zeros(nbins - hist.size, dtype=int))
            hist = clip_histogram(hist, clip_limit)
            hist = map_histogram(hist, 0, NR_OF_GREY - 1, n_pixels)
            map_array[y, x] = hist

    # Interpolate greylevel mappings to get CLAHE image
    ystart = 0
    for y in range(ntiles_y + 1):
        xstart = 0
        if y == 0:  # special case: top row
            ystep = height / 2.0
            yU = 0
            yB = 0
        elif y == ntiles_y:  # special case: bottom row
            ystep = height / 2.0
            yU = ntiles_y - 1
            yB = yU
        else:  # default values
            ystep = height
            yU = y - 1
            yB = yB + 1

        for x in range(ntiles_x + 1):
            if x == 0:  # special case: left column
                xstep = width / 2.0
                xL = 0
                xR = 0
            elif x == ntiles_x:  # special case: right column
                xstep = width / 2.0
                xL = ntiles_x - 1
                xR = xL
            else:  # default values
                xstep = width
                xL = x - 1
                xR = xL + 1

            mapLU = map_array[yU, xL]
            mapRU = map_array[yU, xR]
            mapLB = map_array[yB, xL]
            mapRB = map_array[yB, xR]

            xslice = np.arange(xstart, xstart + xstep)
            yslice = np.arange(ystart, ystart + ystep)
            interpolate(image, xslice, yslice,
                        mapLU, mapRU, mapLB, mapRB, lut)

            xstart += xstep  # set pointer on next matrix */

        ystart += ystep

    if image.shape != orig_shape:
        image = image[:orig_shape[0], :orig_shape[1]]

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

    while n_excess > 0:  # Redistribute remaining excess
        index = 0
        while n_excess > 0 and index < hist.size:
            step_size = int(hist[hist < clip_limit].size / n_excess)
            step_size = max(step_size, 1)
            indices = np.arange(index, hist.size, step_size)
            under = hist[indices] < clip_limit
            hist[under] += 1
            n_excess -= hist[under].size
            index += 1

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


def interpolate(image, xslice, yslice,
                mapLU, mapRU, mapLB, mapRB, lut):
    """Find the new grayscale level for a region using bilinear interpolation.

    Parameters
    ----------
    image : ndarray
        Full image.
    xslice, yslice : array-like
       Indices of the region.
    map* : ndarray
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
    a submatrix of the image. This is done by a bilinear interpolation between
    four different mappings in order to eliminate boundary artifacts.
    """
    norm = xslice.size * yslice.size  # Normalization factor
    # interpolation weight matrices
    x_coef, y_coef = np.meshgrid(np.arange(xslice.size),
                                 np.arange(yslice.size))
    x_inv_coef, y_inv_coef = x_coef[:, ::-1] + 1, y_coef[::-1] + 1

    view = image[int(yslice[0]):int(yslice[-1] + 1),
                 int(xslice[0]):int(xslice[-1] + 1)]
    im_slice = lut[view]
    new = ((y_inv_coef * (x_inv_coef * mapLU[im_slice]
                          + x_coef * mapRU[im_slice])
            + y_coef * (x_inv_coef * mapLB[im_slice]
                        + x_coef * mapRB[im_slice]))
           / norm)
    view[:, :] = new
    return image
