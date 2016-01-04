from libcpp cimport bool

cdef extern from "float.h" nogil:

    float FLT_EPSILON

cdef extern from "math.h":

    double exp(double power)
