"""ctmf.py - constant time per pixel median filtering with an octagonal shape

Reference: S. Perreault and P. Hebert, "Median Filtering in Constant Time",
IEEE Transactions on Image Processing, September 2007.

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky
"""

import warnings
import numpy as np
from . import _ctmf
from ._rank_order import rank_order
from .._shared.utils import deprecated


@deprecated('filter.rank.median')
def median_filter(image, radius=2, mask=None, percent=50):
    """Masked median filter with octagon shape.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    radius : int
        Radius (in pixels) of a circle inscribed into the filtering
        octagon. Must be at least 2. Default radius is 2.
    mask : (M, N) ndarray
        Mask with 1's for significant pixels, 0's for masked pixels.
        By default, all pixels are considered significant.
    percent : int
        The unmasked pixels within the octagon are sorted, and the
        value at `percent` percent of the index range is chosen.
        Default value of 50 gives the median pixel.

    Returns
    -------
    out : (M, N) ndarray
        Filtered array. In areas where the median filter does
        not overlap the mask, the filtered result is undefined, but
        in practice, it will be the lowest value in the valid area.

    Notes
    -----
    Because of the histogram implementation, the number of unique values
    for the output is limited to 256.

    Examples
    --------
    >>> a = np.ones((5, 5))
    >>> a[2, 2] = 10 # introduce outlier
    >>> b = median_filter(a)
    >>> b[2, 2] # the median filter is good at removing outliers
    1.0
    """

    if image.ndim != 2:
        raise TypeError("Input 'image' must be a two-dimensional array.")

    if radius < 2:
        raise ValueError("Input 'radius' must be >= 2.")

    if mask is None:
        mask = np.ones(image.shape, dtype=np.bool)
    mask = np.ascontiguousarray(mask, dtype=np.bool)

    if np.all(~ mask):
        warnings.warn('Mask is all over image! Returning copy of input image.')
        return image.copy()
    
    if (not np.issubdtype(image.dtype, np.int) or
        np.min(image) < 0 or np.max(image) > 255):
        ranked_values, translation = rank_order(image[mask])
        max_ranked_values = np.max(ranked_values)
        if max_ranked_values == 0:
            warnings.warn('Particular case? Returning copy of input image.')
            return image.copy()
        if max_ranked_values > 255:
            ranked_values = ranked_values * 255 // max_ranked_values
        was_ranked = True
    else:
        ranked_values = image[mask]
        was_ranked = False
    ranked_image = np.zeros(image.shape, np.uint8)
    ranked_image[mask] = ranked_values

    mask.dtype = np.uint8
    output = np.zeros(image.shape, np.uint8)

    _ctmf.median_filter(ranked_image, mask, output, radius, percent)
    if was_ranked:
        #
        # The translation gives the original value at each ranking.
        # We rescale the output to the original ranking and then
        # use the translation to look up the original value in the image.
        #
        if max_ranked_values > 255:
            result = translation[output.astype(np.uint32) *
                                 max_ranked_values // 255]
        else:
            result = translation[output]
    else:
        result = output
    return result

