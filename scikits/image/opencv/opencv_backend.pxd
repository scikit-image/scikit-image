import numpy as np
cimport numpy as np
from opencv_type cimport *

cdef extern from "Python.h":
    void Py_INCREF(object)
    
cdef extern from "numpy/arrayobject.h":
    object PyArray_Empty(int, np.npy_intp*, dtype, int)
    
ctypedef np.uint8_t UINT8_t
ctypedef np.int8_t INT8_t
ctypedef np.int16_t INT16_t
ctypedef np.int32_t INT32_t
ctypedef np.float32_t FLOAT32_t
ctypedef np.float64_t FLOAT64_t
           
#-------------------------------------------------------------------------------
# Utility functions for IplImage creation, array validation, etc...
#-------------------------------------------------------------------------------

cdef void populate_iplimage(np.ndarray arr, IplImage* img)
cdef int validate_array(np.ndarray arr) except -1    
cdef int assert_dtype(np.ndarray arr, dtypes) except -1
cdef int assert_ndims(np.ndarray arr, dims) except -1
cdef int assert_nchannels(np.ndarray arr, channels) except -1
cdef int assert_same_dtype(np.ndarray arr1, np.ndarray arr2) except -1
cdef int assert_same_shape(np.ndarray arr1, np.ndarray arr2) except -1
cdef int assert_same_width_and_height(np.ndarray arr1, np.ndarray arr2) except -1
cdef int assert_like(np.ndarray arr1, np.ndarray arr2) except -1
cdef int assert_not_sharing_data(np.ndarray arr1, np.ndarray arr2) except -1
cdef np.ndarray new_array(int ndim, np.npy_intp* shape, dtype)
cdef np.ndarray new_array_like(np.ndarray arr)
cdef np.ndarray new_array_like_diff_dtype(np.ndarray arr, dtype)
cdef np.npy_intp get_array_nbytes(np.ndarray arr)
cdef CvPoint2D32f* array_as_cvPoint2D32f_ptr(np.ndarray arr)
cdef CvTermCriteria get_cvTermCriteria(int, double)
