
cdef inline double nearest_neighbour_interpolation(double* image, int rows,
                                                   int cols, double r,
                                                   double c, char mode,
                                                   double cval)

cdef inline double bilinear_interpolation(double* image, int rows, int cols,
                                          double r, double c, char mode,
                                          double cval)

cdef inline double cubic_interpolation(double x, double[4] f)

cdef inline double bicubic_interpolation(double* image, int rows, int cols,
                                         double r, double c, char mode,
                                         double cval)

cdef inline double get_pixel(double* image, int rows, int cols, int r, int c,
                             char mode, double cval)

cdef inline int coord_map(int dim, int coord, char mode)
