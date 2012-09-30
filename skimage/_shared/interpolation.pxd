
cdef double nearest_neighbour_interpolation(double* image, int rows,
                                            int cols, double r,
                                            double c, char mode,
                                            double cval)

cdef double bilinear_interpolation(double* image, int rows, int cols,
                                   double r, double c, char mode,
                                   double cval)

cdef double quadratic_interpolation(double x, double[3] f)
cdef double biquadratic_interpolation(double* image, int rows, int cols,
                                      double r, double c, char mode,
                                      double cval)

cdef double cubic_interpolation(double x, double[4] f)
cdef double bicubic_interpolation(double* image, int rows, int cols,
                                  double r, double c, char mode,
                                  double cval)

cdef double get_pixel(double* image, int rows, int cols, int r, int c,
                      char mode, double cval)

cdef int coord_map(int dim, int coord, char mode)
