#include <numpy/random/bitgen.h>

void unwrap3D(
        double *wrapped_volume,
        double *unwrapped_volume,
        unsigned char *input_mask,
        int n_k, int n_j, int n_i,
        int wrap_around_k, int wrap_around_j, int wrap_around_i,
        bitgen_t* bitgen_state
        );
