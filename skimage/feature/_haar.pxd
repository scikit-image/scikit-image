cdef struct Point2D:
    int row
    int col


cdef struct Rectangle:
    Point2D top_left
    Point2D bottom_right


cdef Rectangle** _haar_like_feature_coord(int feature_type, int height,
                                          int width, int* n_rectangle,
                                          int* counter_feature)


cpdef haar_like_feature_coord(feature_type, int height, int width)
