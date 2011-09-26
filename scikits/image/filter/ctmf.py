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
from rank_order import rank_order

def median_filter(data, mask=None, radius=1, percent=50):
    '''Masked median filter with octagon shape.

    Parameters
    ----------
    data : (M,N) ndarray, dtype uint8
        Input image.
    mask : (M,N) ndarray, dtype uint8, optional
        A value of 1 indicates a significant pixel, 0
        that a pixel is masked.  By default, all pixels
        are considered.
    radius : {int, 1}, optional
        The radius of a circle inscribed into the filtering
        octagon. Default radius is 1.
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

    if data.ndim!=2:
        raise TypeError("The input 'data' must be a two dimensional array.")

    if mask is None:
        mask = np.ones(data.shape, dtype=np.bool)
    mask = np.ascontiguousarray(mask, dtype=np.bool)

    if np.all(~ mask):
        return data.copy()
    #
    # Normalize the ranked data to 0-255
    #
    if (not np.issubdtype(data.dtype, np.int) or
        np.min(data) < 0 or np.max(data) > 255):
        ranked_data,translation = rank_order(data[mask])
        max_ranked_data = np.max(ranked_data)
        if max_ranked_data == 0:
            return data
        if max_ranked_data > 255:
            ranked_data = ranked_data * 255 // max_ranked_data
        was_ranked = True
    else:
        ranked_data = data[mask]
        was_ranked = False
    input = np.zeros(data.shape, np.uint8 )
    input[mask] = ranked_data

    mask.dtype = np.uint8
    output = np.zeros(data.shape, np.uint8)

    _ctmf.median_filter(input, mask, output, radius, percent)
    if was_ranked:
        #
        # The translation gives the original value at each ranking.
        # We rescale the output to the original ranking and then
        # use the translation to look up the original value in the data.
        #
        if max_ranked_data > 255:
            result = translation[output.astype(np.uint32) * max_ranked_data // 255]
        else:
            result = translation[output]
    else:
        result = output
    return result
