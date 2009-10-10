import ctypes
import numpy as np
cimport numpy as np
from opencv_type cimport *
from opencv_backend import *
from opencv_backend cimport *
from opencv_constants import *

#one of these should work if the user imported the package properly
try:
    cv = ctypes.CDLL('libcv.so')
except:
    try:
        cv = ctypes.CDLL('cv.dll')
    except:
        raise RuntimeError('The opencv libraries were not found. Please make sure they are installed and available on the system path.')


###################################     
# opencv function declarations
###################################

# cvSmooth
ctypedef void (*cvSmoothPtr)(IplImage*, IplImage*, int, int, int, double, double)
cdef cvSmoothPtr c_cvSmooth
c_cvSmooth =  (<cvSmoothPtr*><size_t>ctypes.addressof(cv.cvSmooth))[0]

# cvSobel
ctypedef void (*cvSobelPtr)(IplImage*, IplImage*, int, int, int)
cdef cvSobelPtr c_cvSobel
c_cvSobel = (<cvSobelPtr*><size_t>ctypes.addressof(cv.cvSobel))[0]

# cvLaplace
ctypedef void (*cvLaplacePtr)(IplImage*, IplImage*, int)
cdef cvLaplacePtr c_cvLaplace
c_cvLaplace = (<cvLaplacePtr*><size_t>ctypes.addressof(cv.cvLaplace))[0]

# cvCanny
ctypedef void (*cvCannyPtr)(IplImage*, IplImage*, double, double, int)
cdef cvCannyPtr c_cvCanny
c_cvCanny = (<cvCannyPtr*><size_t>ctypes.addressof(cv.cvCanny))[0]

 

#######################################################################
# Utility Stuff for the C side. Struct creation, error checking, etc..
#######################################################################

####################################
# Function Implementations
####################################
def cvSmooth(np.ndarray src, np.ndarray out=None, int smoothtype=CV_GAUSSIAN, int param1=3,
            int param2=0, double param3=0, double param4=0, bool in_place=False):
    
    validate_array(src)
    if out is not None:
        validate_array(out)
        
    # there are restrictions that must be placed on the data depending on
    # the smoothing operation requested
    
    # CV_BLUR_NO_SCALE
    if smoothtype == CV_BLUR_NO_SCALE:
    
        if in_place:
            raise RuntimeError('In place operation not supported with this filter')
        
        assert_dtype(src, [UINT8, INT8, FLOAT32])                        
        assert_ndims(src, [2])
            
        if out is not None:
            if src.dtype == FLOAT32:
                assert_dtype(out, [FLOAT32])                
            else:
                assert_dtype(out, [INT16])
            assert_same_shape(src, out)                     
        else:
            if src.dtype == FLOAT32:
                out = new_array_like(src)
            else:
                out = new_array_like_diff_dtype(src, INT16)
            
    # CV_BLUR and CV_GAUSSIAN       
    elif smoothtype == CV_BLUR or smoothtype == CV_GAUSSIAN:
        
        assert_dtype(src, [UINT8, INT8, FLOAT32])
        assert_nchannels(src, [1, 3])
        
        if in_place:
            out = src            
        elif out is not None:
            assert_like(src, out)            
        else:
            out = new_array_like(src)
            
    # CV_MEDIAN and CV_BILATERAL
    else: 
        assert_dtype(src, [UINT8, INT8])
        assert_nchannels(src, [1, 3])
        
        if in_place:
            raise RuntimeError('In place operation not supported with this filter')
            
        if out is not None:
            assert_like(src, out)
        else:
            out = new_array_like(src)
    
    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)      
      
    c_cvSmooth(&srcimg, &outimg, smoothtype, param1, param2, param3, param4)        
            
    return out
