import numpy as np
cimport numpy as np

import numpy.ma

cdef extern int unwrap(float* wrapped_image, 
                       float* unwrapped_image, 
                       unsigned char* input_mask, 
                       int image_width, int image_height,
                       int wrap_around_x, int wrap_around_y)

def unwrap2D(input, wrap_around_x = False, wrap_around_y = False):
    
    masked_array = numpy.ma.asarray(input, dtype = np.float32)
    unwrapped_array = _unwrap2D(masked_array.data, 
                                numpy.ma.getmaskarray(masked_array).astype(np.uint8),
                                wrap_around_x, wrap_around_y)
    if numpy.ma.isarray(input):
        return numpy.ma.array(unwrapped_array, mask = input.mask)
    else:
        return unwrapped_array

    #TODO: set_fill to minimum value
    

cdef _unwrap2D(float[:,::1] array, 
               unsigned char[:,::1] mask,
               wrap_around_x, wrap_around_y):
    cdef float[:,::1] unwrapped_array = np.empty_like(array)
    cdef int h = array.shape[0]
    cdef int w = array.shape[1]
    #TODO: check for masked array/
    unwrap(&array[0,0], 
           &unwrapped_array[0,0], 
           &mask[0,0], 
           array.shape[0], array.shape[1],
           wrap_around_x, wrap_around_y,
            )
    return np.asarray(unwrapped_array)
 
    
