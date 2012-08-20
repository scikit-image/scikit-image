cimport numpy as np
import numpy as np


cdef inline double bilinear_interpolation(double* image, int rows, int cols,
                                          double r, double c, char mode,
                                          double cval=*)
