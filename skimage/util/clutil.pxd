cimport cython

cdef int roundUp_int(int size, int multiple)
cdef int[:] roundUp_array(int[:] size, int[:] multiple)
cdef tuple roundUp_tuple(tuple size, tuple multiple)
