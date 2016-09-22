from interpolation cimport coord_map, get_pixel2d
import numpy as np
cimport numpy as cnp


def coord_map_py(Py_ssize_t dim, long coord, mode):
    """Python wrapper for `interpolation.coord_map`."""
    cdef char mode_c = ord(mode[0].upper())
    return coord_map(dim, coord, mode_c)
