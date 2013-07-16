cdef extern void unwrap3D(double* wrapped_volume,
                     double* unwrapped_volume,
                     unsigned char* input_mask,
                     int image_width, int image_height, int volume_depth,
                     int wrap_around_x, int wrap_around_y, int wrap_around_z)

def unwrap_3d(double[:, :, ::1] image,
              unsigned char[:, :, ::1] mask,
              double[:, :, ::1] unwrapped_image,
              wrap_around):
    unwrap3D(&image[0, 0, 0],
             &unwrapped_image[0, 0, 0],
             &mask[0, 0, 0],
             image.shape[2], image.shape[1], image.shape[0], #TODO: check!!!
             wrap_around[2], wrap_around[1], wrap_around[0],
             )
