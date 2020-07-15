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
