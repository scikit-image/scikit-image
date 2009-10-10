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

# cvPreCornerDetect
ctypedef void (*cvPreCorneDetectPtr)(IplImage*, IplImage*, int)
cdef cvPreCorneDetectPtr c_cvPreCornerDetect
c_cvPreCornerDetect = (<cvPreCorneDetectPtr*><size_t>ctypes.addressof(cv.cvPreCornerDetect))[0]

# cvCornerEigenValsAndVecs
ctypedef void (*cvCornerEigenValsAndVecsPtr)(IplImage*, IplImage*, int, int)
cdef cvCornerEigenValsAndVecsPtr c_cvCornerEigenValsAndVecs
c_cvCornerEigenValsAndVecs = (<cvCornerEigenValsAndVecsPtr*><size_t>ctypes.addressof(cv.cvCornerEigenValsAndVecs))[0]

# cvCornerMinEigenVal
ctypedef void (*cvCornerMinEigenValPtr)(IplImage*, IplImage*, int, int)
cdef cvCornerMinEigenValPtr c_cvCornerMinEigenVal
c_cvCornerMinEigenVal = (<cvCornerMinEigenValPtr*><size_t>ctypes.addressof(cv.cvCornerMinEigenVal))[0]

# cvCornerHarris
ctypedef void (*cvCornerHarrisPtr)(IplImage*, IplImage*, int, int, double)
cdef cvCornerHarrisPtr c_cvCornerHarris
c_cvCornerHarris = (<cvCornerHarrisPtr*><size_t>ctypes.addressof(cv.cvCornerHarris))[0]

# cvSmooth
ctypedef void (*cvSmoothPtr)(IplImage*, IplImage*, int, int, int, double, double)
cdef cvSmoothPtr c_cvSmooth
c_cvSmooth =  (<cvSmoothPtr*><size_t>ctypes.addressof(cv.cvSmooth))[0]


####################################
# Function Implementations
####################################
def cvSobel(np.ndarray src, np.ndarray out=None, int xorder=1, int yorder=0,
            int aperture_size=3):
    
    validate_array(src)
    assert_dtype(src, [UINT8, INT8, FLOAT32])
    assert_nchannels(src, [1])
    
    if (aperture_size != 3 and aperture_size != 5 and aperture_size != 7):
        raise ValueError('aperture_size must be 3, 5, or 7')
        
    if out is not None:
        validate_array(out)        
        assert_not_sharing_data(src, out)
        assert_same_shape(src, out)
        assert_nchannels(out, [1])
        if src.dtype == UINT8 or src.dtype == INT8:
            assert_dtype(out, [INT16])
        else:
            assert_dtype(out, [FLOAT32])
    else:
        if src.dtype == UINT8 or src.dtype == INT8:
            out = new_array_like_diff_dtype(src, INT16)
        else:
            out = new_array_like(src)
            
    cdef IplImage srcimg
    cdef IplImage outimg
    
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)
    
    c_cvSobel(&srcimg, &outimg, xorder, yorder, aperture_size)
    
    return out        

def cvLaplace(np.ndarray src, np.ndarray out=None, int aperture_size=3):
    validate_array(src)
    assert_dtype(src, [UINT8, INT8, FLOAT32])
    assert_nchannels(src, [1])
    
    if (aperture_size != 3 and aperture_size != 5 and aperture_size != 7):
        raise ValueError('aperture_size must be 3, 5, or 7')


    if out is not None:
        validate_array(out)   
        assert_not_sharing_data(src, out)
        assert_same_shape(src, out)
        assert_nchannels(out, [1])
        if src.dtype == UINT8 or src.dtype == INT8:
            assert_dtype(out, [INT16])
        else:
            assert_dtype(out, [FLOAT32])
    else:
        if src.dtype == UINT8 or src.dtype == INT8:
            out = new_array_like_diff_dtype(src, INT16)
        else:
            out = new_array_like(src)
            
    cdef IplImage srcimg
    cdef IplImage outimg
    
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)
    
    c_cvLaplace(&srcimg, &outimg, aperture_size)
    
    return out        

def cvCanny(np.ndarray src, np.ndarray out=None, double threshold1=10, 
            double threshold2=50, int aperture_size=3):
    validate_array(src)
    assert_nchannels(src, [1])
    
    if (aperture_size != 3 and aperture_size != 5 and aperture_size != 7):
        raise ValueError('aperture_size must be 3, 5, or 7')

    
    if out is not None:
        validate_array(out)
        assert_nchannels(out, [1])
        assert_same_shape(src, out)
        assert_not_sharing_data(src, out)
    else:
        out = new_array_like(src)
        
    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)
    
    c_cvCanny(&srcimg, &outimg, threshold1, threshold2, aperture_size)
    
    return out

def cvPreCornerDetect(np.ndarray src, np.ndarray out=None, int aperture_size=3):
    validate_array(src)
    assert_dtype(src, [UINT8, FLOAT32])
    assert_nchannels(src, [1])
    
    if (aperture_size != 3 and aperture_size != 5 and aperture_size != 7):
        raise ValueError('aperture_size must be 3, 5, or 7')
        
    if out is not None:
        validate_array(out)
        assert_same_shape(src, out)
        assert_dtype(out, [FLOAT32])
        assert_not_sharing_data(src, out)
    else:
        out = new_array_like_diff_dtype(src, FLOAT32)
        
    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)
    
    c_cvPreCornerDetect(&srcimg, &outimg, aperture_size)
    
    return out    
    
def cvCornerEigenValsAndVecs(np.ndarray src, int block_size=3, 
                                             int aperture_size=3):
                             
    # no option for the out argument on this one. Its easier just 
    # to make it for them as there is only 1 valid out array for any 
    # given source array
    
    validate_array(src)
    assert_nchannels(src, [1])
    assert_dtype(src, [UINT8, FLOAT32])
    
    if (aperture_size != 3 and aperture_size != 5 and aperture_size != 7):
        raise ValueError('aperture_size must be 3, 5, or 7')
    
    cdef np.npy_intp outshape[2]    
    outshape[0] = src.shape[0]
    outshape[1] = src.shape[1] * <np.npy_intp>6    
    
    out = new_array(2, outshape, FLOAT32)
    
    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)    
    
    c_cvCornerEigenValsAndVecs(&srcimg, &outimg, block_size, aperture_size)
    
    return out.reshape(out.shape[0], -1, 6)
    
def cvCornerMinEigenVal(np.ndarray src, int block_size=3, 
                                        int aperture_size=3):
                                        
    # no option for the out argument on this one. Its easier just 
    # to make it for them as there is only 1 valid out array for any 
    # given source array
    
    validate_array(src)
    assert_nchannels(src, [1])
    assert_dtype(src, [UINT8, FLOAT32])
    
    if (aperture_size != 3 and aperture_size != 5 and aperture_size != 7):
        raise ValueError('aperture_size must be 3, 5, or 7')
    
    out = new_array_like_diff_dtype(src, FLOAT32)
    
    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)    
    
    c_cvCornerMinEigenVal(&srcimg, &outimg, block_size, aperture_size)
    
    return out

def cvCornerHarris(np.ndarray src, int block_size=3, int aperture_size=3,
                                                     double k=0.04):
                                            
    # no option for the out argument on this one. Its easier just 
    # to make it for them as there is only 1 valid out array for any 
    # given source array
    
    validate_array(src)
    assert_nchannels(src, [1])
    assert_dtype(src, [UINT8, FLOAT32])
    
    if (aperture_size != 3 and aperture_size != 5 and aperture_size != 7):
        raise ValueError('aperture_size must be 3, 5, or 7')
    
    out = new_array_like_diff_dtype(src, FLOAT32)
    
    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)    
    
    c_cvCornerHarris(&srcimg, &outimg, block_size, aperture_size, k)
    
    return out    


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
