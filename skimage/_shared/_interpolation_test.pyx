from interpolation cimport coord_map as _coord_map

def coord_map(Py_ssize_t dim, long coord, mode):
    cdef char mode_c = ord(mode[0].upper())
    return _coord_map(dim, coord, mode_c)