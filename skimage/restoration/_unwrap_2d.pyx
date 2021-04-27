# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False


cdef extern from "unwrap_2d_ljmu.h":
    void unwrap2D(
            double *wrapped_image,
            double *UnwrappedImage,
            unsigned char *input_mask,
            int image_width, int image_height,
            int wrap_around_x, int wrap_around_y,
            char use_seed, unsigned int seed
            ) nogil


def unwrap_2d(double[:, ::1] image,
              unsigned char[:, ::1] mask,
              double[:, ::1] unwrapped_image,
              wrap_around,
              seed):
    cdef:
        unsigned int cseed
        char use_seed
        int wrap_around_x
        int wrap_around_y

    # convert from python types to C types so we can release the GIL
    use_seed = seed is None
    cseed = 0 if seed is None else seed
    wrap_around_y, wrap_around_x = wrap_around
    with nogil:
        unwrap2D(&image[0, 0],
                 &unwrapped_image[0, 0],
                 &mask[0, 0],
                 image.shape[1], image.shape[0],
                 wrap_around_x, wrap_around_y,
                 use_seed, cseed)
