cdef extern from *:
    void unwrap2D(double* wrapped_image,
                  double* unwrapped_image,
                  unsigned char* input_mask,
                  int image_width, int image_height,
                  int wrap_around_x, int wrap_around_y,
                  char use_seed, unsigned int seed)

def unwrap_2d(double[:, ::1] image,
              unsigned char[:, ::1] mask,
              double[:, ::1] unwrapped_image,
              wrap_around,
              seed):
    unwrap2D(&image[0, 0],
             &unwrapped_image[0, 0],
             &mask[0, 0],
             image.shape[1], image.shape[0],
             wrap_around[1], wrap_around[0],
             seed is None, 0 if seed is None else seed)
