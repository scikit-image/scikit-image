#include <numpy/random/bitgen.h>

void unwrap2D(
        double *wrapped_image,
        double *UnwrappedImage,
        unsigned char *input_mask,
        int image_width, int image_height,
        int wrap_around_x, int wrap_around_y,
        bitgen_t* bitgen_state
        );
