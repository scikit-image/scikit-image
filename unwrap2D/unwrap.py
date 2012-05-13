import numpy as np
from unwrap2D import _unwrap2D

def unwrap2D(wrapped_array, wrap_around_x = False, wrap_around_y = False):
    wrapped_array = np.require(wrapped_array, np.float32, ['C'])
    wrapped_array_masked = np.ma.asarray(wrapped_array)
    unwrapped_array = np.empty_like(wrapped_array_masked.data)
    
    _unwrap2D(wrapped_array_masked.data, 
              np.ma.getmaskarray(wrapped_array_masked).astype(np.uint8),
              unwrapped_array,
              wrap_around_x, wrap_around_y)
    if np.ma.isMaskedArray(wrapped_array):
        return np.ma.array(unwrapped_array, mask = wrapped_array_masked.mask)
    else:
        return unwrapped_array

    #TODO: set_fill to minimum value
    #TODO: check for empty mask, not a single contiguous pixel

