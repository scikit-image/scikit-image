cdef extern unwrap2D(float* wrapped_image, 
                     float* unwrapped_image, 
                     unsigned char* input_mask, 
                     int image_width, int image_height,
                     int wrap_around_x, int wrap_around_y)

def _unwrap2D(float[:,::1] array, 
              unsigned char[:,::1] mask,
              float[:,::1] unwrapped_array,
              wrap_around_x, wrap_around_y):
    cdef int h = array.shape[0]
    cdef int w = array.shape[1]
    unwrap2D(&array[0,0], 
             &unwrapped_array[0,0], 
             &mask[0,0], 
             array.shape[1], array.shape[0],
             wrap_around_x, wrap_around_y,
             )
 
    
