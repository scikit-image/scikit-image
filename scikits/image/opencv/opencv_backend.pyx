import numpy as np
cimport numpy as np
from opencv_constants import *
from opencv_type cimport *


np.import_array()

#-------------------------------------------------------------------------------
# Data Type Handling
#-------------------------------------------------------------------------------

# for some reason these have to declared as dtype objects rather than just the 
# dtype itself....
UINT8 = np.dtype('uint8') 
INT8 = np.dtype('int8') 
INT16 = np.dtype('int16')
INT32 = np.dtype('int32')
FLOAT32 = np.dtype('float32')
FLOAT64 = np.dtype('float64')

cdef int IPL_DEPTH_SIGN = 0x80000000
cdef int IPL_DEPTH_8U = 8
cdef int IPL_DEPTH_8S = (IPL_DEPTH_SIGN | 8)
cdef int IPL_DEPTH_16S = (IPL_DEPTH_SIGN | 16)
cdef int IPL_DEPTH_32S = (IPL_DEPTH_SIGN | 32)
cdef int IPL_DEPTH_32F = 32
cdef int IPL_DEPTH_64F = 64


# I'd like a better to associate the IPL data type flag to the proper numpy 
# types without using a dictionary.
_ipltypes = {UINT8: IPL_DEPTH_8U, INT8: IPL_DEPTH_8S, INT16: IPL_DEPTH_16S, 
             INT32: IPL_DEPTH_32S, FLOAT32: IPL_DEPTH_32F, 
             FLOAT64: IPL_DEPTH_64F}      

           
#-------------------------------------------------------------------------------
# Utility functions for IplImage creation, array validation, etc...
#-------------------------------------------------------------------------------


cdef int IPLIMAGE_SIZE = sizeof(IplImage)

cdef void populate_iplimage(np.ndarray arr, IplImage* img):
    # The numpy array should be validated with the validate_array
    # function before using this function.
    # This function assumes that the array has successfully passed
    # validation
    
    # everything that will never change
    img.nSize = IPLIMAGE_SIZE
    img.ID = 0
    img.dataOrder = 0
    img.origin = 0
    img.roi = NULL
    img.maskROI = NULL
    img.imageId = NULL
    img.tileInfo = NULL
    
    cdef int channels
    cdef int ndim = arr.ndim
    cdef np.npy_intp* shape = arr.shape
    cdef np.npy_intp* strides = arr.strides   
            
    # nChannels is essentially the value of np.shape[2] of a 3D numpy array
    # for a 2D array, nChannels is 1
    if ndim == 2:
        img.nChannels = 1
    else:
        img.nChannels = shape[2]
       
    img.depth = _ipltypes[arr.dtype]
    img.width = shape[1]
    img.height = shape[0]    
    img.imageSize = arr.nbytes
    img.imageData = <char*>arr.data
    img.widthStep = strides[0]
    
    # really doesn't matter what this is set to, because opencv only uses it to 
    # deallocate images, but it will never attempt to deallocate images we 
    # create ourselves. 
    img.imageDataOrigin = <char*>NULL

cdef int validate_array(np.ndarray arr) except -1:
    if arr.ndim != 2 and arr.ndim != 3:
        raise ValueError('Arrays must have either 2 or 3 dimensions')
    if arr.ndim == 3:
        if arr.shape[2] > 4:
            raise ValueError('A 3D array must have 4 or less channels')
    if arr.dtype not in _ipltypes:
        raise ValueError('Arrays must have one of the following dtypes: uint8, int8, int16, int32, float32, float64')        
    return 1
    
cdef int assert_dtype(np.ndarray arr, dtypes) except -1:
    if arr.dtype not in dtypes:
        raise ValueError('Unsupported dtype for this operation. \
                          Supported dtypes are %s' % str(dtypes))
    return 1
    
cdef int assert_ndims(np.ndarray arr, dims) except -1:
    if arr.ndim not in dims:
        raise ValueError('Incorrect number of dimensions')
    return 1
    
cdef int assert_nchannels(np.ndarray arr, channels) except -1:
    cdef int nchannels
    if arr.ndim == 2:
        nchannels = 1
    else:
        nchannels = arr.shape[2]    
    if nchannels not in channels:
        raise ValueError('Incorrect number of channels')
    return 1
                 
cdef int assert_same_dtype(np.ndarray arr1, np.ndarray arr2) except -1:
    if arr1.dtype != arr2.dtype:
        raise ValueError('dtypes not same')
    return 1
    
cdef int assert_same_shape(np.ndarray arr1, np.ndarray arr2) except -1:
    if not np.PyArray_SAMESHAPE(arr1, arr2):
        raise ValueError('arrays not same shape')
    return 1
    
cdef int assert_same_width_and_height(np.ndarray arr1, np.ndarray arr2) except -1:
    cdef np.npy_intp* shape1 = arr1.shape
    cdef np.npy_intp* shape2 = arr2.shape
    if (shape1[0] != shape2[0]) or (shape1[1] != shape2[1]):
        raise ValueError('Arrays must have same width and height')
    return 1
      
cdef int assert_like(np.ndarray arr1, np.ndarray arr2) except -1:
    assert_same_dtype(arr1, arr2)
    assert_same_shape(arr1, arr2)
    return 1

cdef int assert_not_sharing_data(np.ndarray arr1, np.ndarray arr2) except -1:
    if arr1.data == arr2.data:
        raise ValueError('In place operation not supported. Make sure \
                          the out array is not just a view of src array')
    return 1
              
cdef np.ndarray new_array(int ndim, np.npy_intp* shape, dtype):
    # need to incref because numpy will apprently steal a dtype reference
    Py_INCREF(<object>dtype)
    return PyArray_Empty(ndim, shape, dtype, 0)

cdef np.ndarray new_array_like(np.ndarray arr):
    # need to incref because numpy will apprently steal a dtype reference
    Py_INCREF(<object>arr.dtype)
    return PyArray_Empty(arr.ndim, arr.shape, arr.dtype, 0)
        
cdef np.ndarray new_array_like_diff_dtype(np.ndarray arr, dtype):
    # need to incref because numpy will apprently steal a dtype reference    
    Py_INCREF(<object>dtype)
    return PyArray_Empty(arr.ndim, arr.shape, dtype, 0)

cdef CvPoint2D32f* array_as_cvPoint2D32f_ptr(np.ndarray arr):
    cdef CvPoint2D32f* point2Darr   
    point2Darr = <CvPoint2D32f*>arr.data
    return point2Darr
    
cdef np.npy_intp get_array_nbytes(np.ndarray arr):
    cdef np.npy_intp nbytes = np.PyArray_NBYTES(arr)
    return nbytes
    
cdef CvTermCriteria get_cvTermCriteria(int iterations, double epsilon):
    cdef CvTermCriteria crit    
    if iterations and epsilon:
        crit.type = <int>(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS)
        crit.max_iter = iterations
        crit.epsilon = epsilon
    elif iterations and not epsilon:
        crit.type = <int>CV_TERMCRIT_ITER
        crit.max_iter = iterations
        crit.epsilon = 0.
    else:
        crit.type = <int>CV_TERMCRIT_EPS
        crit.max_iter = 0
        crit.epsilon = epsilon
    return crit
    

       
        
    


