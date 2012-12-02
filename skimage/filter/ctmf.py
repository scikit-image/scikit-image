'''ctmf.py - constant time per pixel median filtering with an octagonal shape

Reference: S. Perreault and P. Hebert, "Median Filtering in Constant Time",
IEEE Transactions on Image Processing, September 2007.

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky
'''

import numpy as np
from . import _ctmf
from ._rank_order import rank_order


def median_filter(image, radius=2, mask=None, percent=50):
    '''Masked median filter with octagon shape.

    Parameters
    ----------
    image : (M,N) ndarray, dtype uint8
        Input image.
    radius : {int, 2}, optional
        The radius of a circle inscribed into the filtering
        octagon. Must be at least 2.  Default radius is 2.
    mask : (M,N) ndarray, dtype uint8, optional
        A value of 1 indicates a significant pixel, 0
        that a pixel is masked.  By default, all pixels
        are considered.
    percent : {int, 50}, optional
        The unmasked pixels within the octagon are sorted, and the
        value at the `percent`-th index chosen.  For example, the
        default value of 50 chooses the median pixel.

    Returns
    -------
    out : (M,N) ndarray, dtype uint8
        Filtered array.  In areas where the median filter does
        not overlap the mask, the filtered result is underfined, but
        in practice, it will be the lowest value in the valid area.

    '''

    if image.ndim != 2:
        raise TypeError("The input 'image' must be a two dimensional array.")

    if radius < 2:
        raise ValueError("The input 'radius' must be >= 2.")

    if mask is None:
        mask = np.ones(image.shape, dtype=np.bool)
    mask = np.ascontiguousarray(mask, dtype=np.bool)

    if np.all(~ mask):
        return image.copy()
    #
    # Normalize the ranked image to 0-255
    #
    if (not np.issubdtype(image.dtype, np.int) or
        np.min(image) < 0 or np.max(image) > 255):
        ranked_image, translation = rank_order(image[mask])
        max_ranked_image = np.max(ranked_image)
        if max_ranked_image == 0:
            return image
        if max_ranked_image > 255:
            ranked_image = ranked_image * 255 // max_ranked_image
        was_ranked = True
    else:
        ranked_image = image[mask]
        was_ranked = False
    input = np.zeros(image.shape, np.uint8)
    input[mask] = ranked_image

    mask.dtype = np.uint8
    output = np.zeros(image.shape, np.uint8)

    _ctmf.median_filter(input, mask, output, radius, percent)
    if was_ranked:
        #
        # The translation gives the original value at each ranking.
        # We rescale the output to the original ranking and then
        # use the translation to look up the original value in the image.
        #
        if max_ranked_image > 255:
            result = translation[output.astype(np.uint32) *
                                 max_ranked_image // 255]
        else:
            result = translation[output]
    else:
        result = output
    return result
