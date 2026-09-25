#include "unwrap_common.h"

int unwrap3D(
        double *wrapped_volume,
        double *unwrapped_volume,
        unsigned char *input_mask,
        intptr_t n_k, intptr_t n_j, intptr_t n_i,
        int wrap_around_k, int wrap_around_j, int wrap_around_i,
        bitgen_t* bitgen_state
        );
