'''ctmf.py - constant time per pixel median filtering with an octagonal shape

Reference: S. Perreault and P. Hebert, "Median Filtering in Constant Time",
IEEE Transactions on Image Processing, September 2007.

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentstky
'''

import numpy as np
import _ctmf
from rank_order import rank_order

def median_filter(data, mask, radius, percent=50):
    '''Masked median filter with octagonal shape
    
    data - array of data to be median filtered.
    mask - mask of significant pixels in data
    radius - the radius of a circle inscribed into the filtering octagon
    percent - conceptually, order the significant pixels in the octagon,
              count them and choose the pixel indexed by the percent
              times the count divided by 100. More simply, 50 = median
    returns a filtered array.  In areas where the median filter does
      not overlap the mask, the filtered result is undefined, but in
      practice, it will be the lowest value in the valid area.
    '''
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
            ranked_data = ranked_data * 255 / max_ranked_data
        was_ranked = True
    else:
        ranked_data = data[mask]
        was_ranked = False
    input = np.zeros(data.shape, np.uint8 )
    input[mask] = ranked_data
    
    mmask = np.ascontiguousarray(mask, np.uint8)
    
    output = np.zeros(data.shape, np.uint8)
    
    _ctmf.median_filter(input, mmask, output, radius, percent)
    if was_ranked:
        #
        # The translation gives the original value at each ranking.
        # We rescale the output to the original ranking and then
        # use the translation to look up the original value in the data.
        #
        if max_ranked_data > 255:
            result = translation[output.astype(np.uint32) * max_ranked_data / 255]
        else:
            result = translation[output]
    else:
        result = output
    return result

