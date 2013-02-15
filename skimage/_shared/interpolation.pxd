
cdef double nearest_neighbour_interpolation(double* image, Py_ssize_t rows,
                                            Py_ssize_t cols, double r,
                                            double c, char mode,
                                            double cval)

cdef double bilinear_interpolation(double* image, Py_ssize_t rows, Py_ssize_t cols,
                                   double r, double c, char mode,
                                   double cval)

cdef double quadratic_interpolation(double x, double[3] f)
cdef double biquadratic_interpolation(double* image, Py_ssize_t rows, Py_ssize_t cols,
                                      double r, double c, char mode,
                                      double cval)

cdef double cubic_interpolation(double x, double[4] f)
cdef double bicubic_interpolation(double* image, Py_ssize_t rows, Py_ssize_t cols,
                                  double r, double c, char mode,
                                  double cval)

cdef double get_pixel2d(double* image, Py_ssize_t rows, Py_ssize_t cols, Py_ssize_t r,
                        Py_ssize_t c, char mode, double cval)

cdef double get_pixel3d(double* image, Py_ssize_t rows, Py_ssize_t cols, Py_ssize_t dims,
                        Py_ssize_t r, Py_ssize_t c, Py_ssize_t d, char mode, double cval)

cdef Py_ssize_t coord_map(Py_ssize_t dim, Py_ssize_t coord, char mode)
