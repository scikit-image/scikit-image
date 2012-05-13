cdef extern unwrap3D(float* wrapped_volume, 
                     float* unwrapped_volume, 
                     unsigned char* input_mask, 
                     int image_width, int image_height, int volume_depth,
                     int wrap_around_x, int wrap_around_y, int wrap_around_z)

def _unwrap3D(float[:,:,::1] array, 
              unsigned char[:,:,::1] mask,
              float[:,:,::1] unwrapped_array,
              wrap_around_x, wrap_around_y, wrap_around_z):
    unwrap3D(&array[0,0,0], 
             &unwrapped_array[0,0,0], 
             &mask[0,0,0], 
             array.shape[0], array.shape[1], array.shape[2], #TODO: check!!!
             wrap_around_x, wrap_around_y, wrap_around_z,
             )
 
    
