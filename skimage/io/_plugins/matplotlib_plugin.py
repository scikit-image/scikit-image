import numpy as np
import warnings
import matplotlib.pyplot as plt
from skimage.util import dtype as dtypes


def imshow(im, *args, **kwargs):
    """Show the input image and return the current axes.

    Images are assumed to have standard range for their type. For
    example, if a floating point image has values in [0, 0.5], the
    most intense color will be gray50, not white, as would be the
    default in matplotlib.

    In contrast, if the image exceeds the standard range, this
    function defaults back to displaying exactly the range of the
    input image.

    Parameters
    ----------
    im : array, shape (M, N[, 3])
        The image to display.

    *args, **kwargs : positional and keyword arguments
        These are passed directly to `matplotlib.pyplot.imshow`.

    Returns
    -------
    ax : `matplotlib.pyplot.Axes`
        The axes showing the image.
    """
    if plt.gca().has_data():
        plt.figure()
    immin, immax = np.min(im), np.max(im)
    supported_dtype = im.dtype in dtypes._supported_types
    if supported_dtype:
        lo, hi = dtypes.dtype_range[im.dtype.type]
        if immin >= 0:
            # display range starts at 0 for nonnegative images
            lo = 0
    else:
        warnings.warn("Non-standard image type; displaying image with "
                      "stretched contrast.")
    out_of_range_float = (np.issubdtype(im.dtype, np.float) and
                          (immin < lo or immax > hi))
    if out_of_range_float:
        warnings.warn("Float image out of standard range; displaying image "
                      "with stretched contrast.")
    low_dynamic_range = (immin != immax and
                         (float(immax - immin) / (hi - lo)) < (1. / 255))
    if low_dynamic_range:
        warnings.warn("Low image dynamic range; displaying image with "
                      "stretched contrast.")
    if not supported_dtype or out_of_range_float or low_dynamic_range:
        lo, hi = immin, immax
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('cmap', 'gray')
    kwargs.setdefault('vmin', lo)
    kwargs.setdefault('vmax', hi)
    ax = plt.imshow(im, *args, **kwargs)
    if not supported_dtype or out_of_range_float or low_dynamic_range:
        ax.colorbar()
    return ax

imread = plt.imread
show = plt.show


def _app_show():
    show()
