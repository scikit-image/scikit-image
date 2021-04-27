# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False


cdef extern from "unwrap_3d_ljmu.h":
    void unwrap3D(
            double *wrapped_volume,
            double *unwrapped_volume,
            unsigned char *input_mask,
            int volume_width, int volume_height, int volume_depth,
            int wrap_around_x, int wrap_around_y, int wrap_around_z,
            char use_seed, unsigned int seed
            ) nogil



def unwrap_3d(double[:, :, ::1] image,
              unsigned char[:, :, ::1] mask,
              double[:, :, ::1] unwrapped_image,
              wrap_around,
              seed):
    cdef:
        unsigned int cseed
        char use_seed
        int wrap_around_x
        int wrap_around_y
        int wrap_around_z

    # convert from python types to C types so we can release the GIL
    use_seed = seed is None
    cseed = 0 if seed is None else seed
    wrap_around_z, wrap_around_y, wrap_around_x = wrap_around

    with nogil:
        unwrap3D(&image[0, 0, 0],
                 &unwrapped_image[0, 0, 0],
                 &mask[0, 0, 0],
                 image.shape[2], image.shape[1], image.shape[0], #TODO: check!!!
                 wrap_around_x, wrap_around_y, wrap_around_z,
                 use_seed, cseed)
