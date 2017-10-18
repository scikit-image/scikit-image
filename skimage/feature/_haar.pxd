cimport cython
cimport numpy as np


ctypedef fused integral_floating:
    cython.integral
    cython.floating


cdef struct Point2D:
    int row
    int col


cdef struct Rectangle:
    Point2D top_left
    Point2D bottom_right


cdef inline void set_rectangle_feature(Rectangle* rectangle,
                                       int top_y, int top_x,
                                       int bottom_y, int bottom_x) nogil:
    rectangle[0].top_left.row = top_y
    rectangle[0].top_left.col = top_x
    rectangle[0].bottom_right.row = bottom_y
    rectangle[0].bottom_right.col = bottom_x


cdef Rectangle** _haar_like_feature_coord(int feature_type, int height,
                                          int width, int* n_rectangle,
                                          int* counter_feature) nogil


cpdef haar_like_feature_coord(feature_type, int height, int width)


cdef integral_floating[:, ::1] _haar_like_feature(
    integral_floating[:, ::1] roi_ii,
    Rectangle** coord,
    int n_rectangle, int n_feature)


cpdef haar_like_feature(integral_floating[:, ::1] roi_ii, feature_type)
