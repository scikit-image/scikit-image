# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

cimport numpy as cnp


cdef extern from "unwrap_3d_ljmu.h":
    void unwrap3D(
            cnp.float64_t *wrapped_volume,
            cnp.float64_t *unwrapped_volume,
            unsigned char *input_mask,
            int volume_width, int volume_height, int volume_depth,
            int wrap_around_x, int wrap_around_y, int wrap_around_z,
            unsigned int seed
            ) noexcept nogil



def unwrap_3d(cnp.float64_t[:, :, ::1] image,
              unsigned char[:, :, ::1] mask,
              cnp.float64_t[:, :, ::1] unwrapped_image,
              wrap_around,
              unsigned int seed):
    cdef:
        unsigned int cseed
        int wrap_around_x
        int wrap_around_y
        int wrap_around_z

    # Convert from python types to C types so we can release the GIL
    wrap_around_z, wrap_around_y, wrap_around_x = wrap_around

    with nogil:
        unwrap3D(&image[0, 0, 0],
                 &unwrapped_image[0, 0, 0],
                 &mask[0, 0, 0],
                 image.shape[2], image.shape[1], image.shape[0], #TODO: check!!!
                 wrap_around_x, wrap_around_y, wrap_around_z,
                 seed)
