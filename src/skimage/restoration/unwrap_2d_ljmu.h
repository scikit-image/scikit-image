#include <numpy/random/bitgen.h>

void unwrap2D(
        double *wrapped_image,
        double *UnwrappedImage,
        unsigned char *input_mask,
        int n_j, int n_i,
        int wrap_around_j, int wrap_around_i,
        bitgen_t* bitgen_state
        );
