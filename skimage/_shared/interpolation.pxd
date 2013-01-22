
cdef double nearest_neighbour_interpolation(double* image, ssize_t rows,
                                            ssize_t cols, double r,
                                            double c, char mode,
                                            double cval)

cdef double bilinear_interpolation(double* image, ssize_t rows, ssize_t cols,
                                   double r, double c, char mode,
                                   double cval)

cdef double quadratic_interpolation(double x, double[3] f)
cdef double biquadratic_interpolation(double* image, ssize_t rows, ssize_t cols,
                                      double r, double c, char mode,
                                      double cval)

cdef double cubic_interpolation(double x, double[4] f)
cdef double bicubic_interpolation(double* image, ssize_t rows, ssize_t cols,
                                  double r, double c, char mode,
                                  double cval)

cdef double get_pixel2d(double* image, ssize_t rows, ssize_t cols, ssize_t r,
                        ssize_t c, char mode, double cval)

cdef double get_pixel3d(double* image, ssize_t rows, ssize_t cols, ssize_t dims,
                        ssize_t r, ssize_t c, ssize_t d, char mode, double cval)

cdef ssize_t coord_map(ssize_t dim, ssize_t coord, char mode)
