import numpy as np
cimport numpy as np

cdef extern int unwrap(float* WrappedImage, float* UnwrappedImage, unsigned char* input_mask, 
                       int image_width, int image_height)

def unwrap2D(float[:,::1] array, unsigned char[:,::1] mask):
    cdef float[:,::1] unwrapped_array = np.empty_like(array)
    cdef int h = array.shape[0]
    cdef int w = array.shape[1]
    unwrap(&array[0,0], 
           &unwrapped_array[0,0], 
           &mask[0,0], 
           array.shape[0], array.shape[1])
    return unwrapped_array
 
    
