# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer

cimport numpy as cnp
from numpy.random cimport bitgen_t


cdef extern from "unwrap_3d_ljmu.h":
    void unwrap3D(
            cnp.float64_t *wrapped_volume,
            cnp.float64_t *unwrapped_volume,
            unsigned char *input_mask,
            int volume_width, int volume_height, int volume_depth,
            int wrap_around_x, int wrap_around_y, int wrap_around_z,
            bitgen_t* bitgen_state
            ) noexcept nogil


def unwrap_3d(cnp.float64_t[:, :, ::1] image,
              unsigned char[:, :, ::1] mask,
              cnp.float64_t[:, :, ::1] unwrapped_image,
              wrap_around,
              rng):
    cdef:
        int wrap_around_x
        int wrap_around_y
        int wrap_around_z
        object bitgen, capsule
        const char* capsule_name
        bitgen_t* bitgen_state

    # Convert from python types to C types so we can release the GIL
    wrap_around_z, wrap_around_y, wrap_around_x = wrap_around
    bitgen = getattr(rng, 'bit_generator', rng)
    capsule = bitgen.capsule
    capsule_name = "BitGenerator"

    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid BitGenerator capsule.")
    bitgen_state = <bitgen_t *>PyCapsule_GetPointer(capsule, capsule_name)

    with bitgen.lock, nogil:
        unwrap3D(&image[0, 0, 0],
                 &unwrapped_image[0, 0, 0],
                 &mask[0, 0, 0],
                 image.shape[2], image.shape[1], image.shape[0], #TODO: check!!!
                 wrap_around_x, wrap_around_y, wrap_around_z,
                 bitgen_state)
