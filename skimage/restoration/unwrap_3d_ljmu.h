void unwrap3D(
        double *wrapped_volume,
        double *unwrapped_volume,
        unsigned char *input_mask,
        int volume_width, int volume_height, int volume_depth,
        int wrap_around_x, int wrap_around_y, int wrap_around_z,
        char use_seed, unsigned int seed
        );

