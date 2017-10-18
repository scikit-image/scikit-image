cimport numpy as cnp


ctypedef fused integral_floating:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cnp.float32_t
    cnp.float64_t


cdef struct Point2D:
    Py_ssize_t row
    Py_ssize_t col


cdef struct Rectangle:
    Point2D top_left
    Point2D bottom_right


cdef inline void set_rectangle_feature(Rectangle* rectangle,
                                       Py_ssize_t top_y,
                                       Py_ssize_t top_x,
                                       Py_ssize_t bottom_y,
                                       Py_ssize_t bottom_x) nogil:
    rectangle[0].top_left.row = top_y
    rectangle[0].top_left.col = top_x
    rectangle[0].bottom_right.row = bottom_y
    rectangle[0].bottom_right.col = bottom_x


cdef Rectangle** _haar_like_feature_coord(unsigned int feature_type,
                                          Py_ssize_t height,
                                          Py_ssize_t width,
                                          Py_ssize_t* n_rectangle,
                                          Py_ssize_t* counter_feature) nogil


cpdef haar_like_feature_coord(feature_type, height, width)


cdef integral_floating[:, ::1] _haar_like_feature(
    integral_floating[:, ::1] roi_ii,
    Rectangle** coord,
    Py_ssize_t n_rectangle, Py_ssize_t n_feature)


cpdef haar_like_feature(integral_floating[:, ::1] roi_ii, feature_type)
