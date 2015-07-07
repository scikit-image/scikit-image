from interpolation cimport coord_map as _coord_map
from interpolation cimport get_pixel2d
import numpy as np
cimport numpy as cnp


def coord_map(Py_ssize_t dim, long coord, mode):
    """ interpolation.coord_map python wrapper """
    cdef char mode_c = ord(mode[0].upper())
    return _coord_map(dim, coord, mode_c)


def extend_image(image, pad=10, mode='C', cval=0):
    """ can be used to verify proper get_pixel2d behavior. """
    cdef:
        Py_ssize_t rows = image.shape[0]
        Py_ssize_t cols = image.shape[1]
        long ro, co
        char mode_c = ord(mode[0].upper())

    image = np.ascontiguousarray(image.astype(np.float64))
    output_shape = np.asarray(image.shape) + 2*pad
    image_out = np.zeros(output_shape, dtype=image.dtype)
    for r in range(-pad, rows+pad):
        for c in range(-pad, cols+pad):
            ro = r + pad
            co = c + pad
            image_out[ro, co] = get_pixel2d(<double*> cnp.PyArray_DATA(image),
                                          rows, cols, <long> r, <long> c,
                                          mode_c, <double> cval)
    return image_out
