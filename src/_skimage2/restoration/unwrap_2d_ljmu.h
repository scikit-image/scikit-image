#include "unwrap_common.h"

int unwrap2D(
        const double *wrapped_image,
        double *UnwrappedImage,
        const unsigned char *input_mask,
        intptr_t n_j, intptr_t n_i,
        int wrap_around_j, int wrap_around_i,
        bitgen_t* bitgen_state
        );
