from .._shared.fused_numerics cimport np_floats

cpdef int _multiblock_lbp(np_floats[:, ::1] int_image,
                          Py_ssize_t r,
                          Py_ssize_t c,
                          Py_ssize_t width,
                          Py_ssize_t height) nogil
