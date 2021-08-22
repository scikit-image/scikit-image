from .fused_numerics cimport np_floats


cdef enum:
    OUTSIDE = 0
    INSIDE  = 1
    VERTEX  = 2
    EDGE    = 3


cdef unsigned char point_in_polygon(np_floats[::1] xp, np_floats[::1] yp,
                                    np_floats x, np_floats y) nogil

cdef void points_in_polygon(np_floats[::1] xp, np_floats[::1] yp,
                            np_floats[::1] x, np_floats[::1] y,
                            unsigned char[::1] result) nogil
