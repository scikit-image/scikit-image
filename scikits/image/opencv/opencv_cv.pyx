import ctypes
import numpy as np
cimport numpy as np
from python cimport *
#from stdlib cimport *
from opencv_type cimport *
from opencv_backend import *
from opencv_backend cimport *
from opencv_constants import *

from opencv_constants import *

# Without the opencv libraries, this extension module cannot function,
# so we raise an exception if loading fails.
#
# Note, however, that users should be able to import scikits.image.opencv
# itself without having any of the libraries installed
# (the opencv functionality is then simply not available)
#
from _libimport import cv
if cv is None:
    raise RuntimeError('Could not load OpenCV libraries.')

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
c_cvPreCornerDetect = (<cvPreCorneDetectPtr*><size_t>
                       ctypes.addressof(cv.cvPreCornerDetect))[0]

# cvCornerEigenValsAndVecs
ctypedef void (*cvCornerEigenValsAndVecsPtr)(IplImage*, IplImage*, int, int)
cdef cvCornerEigenValsAndVecsPtr c_cvCornerEigenValsAndVecs
c_cvCornerEigenValsAndVecs = (<cvCornerEigenValsAndVecsPtr*><size_t>
                              ctypes.addressof(cv.cvCornerEigenValsAndVecs))[0]

# cvCornerMinEigenVal
ctypedef void (*cvCornerMinEigenValPtr)(IplImage*, IplImage*, int, int)
cdef cvCornerMinEigenValPtr c_cvCornerMinEigenVal
c_cvCornerMinEigenVal = (<cvCornerMinEigenValPtr*><size_t>
                         ctypes.addressof(cv.cvCornerMinEigenVal))[0]

# cvCornerHarris
ctypedef void (*cvCornerHarrisPtr)(IplImage*, IplImage*, int, int, double)
cdef cvCornerHarrisPtr c_cvCornerHarris
c_cvCornerHarris = (<cvCornerHarrisPtr*><size_t>
                    ctypes.addressof(cv.cvCornerHarris))[0]

# cvFindCornerSubPix
ctypedef void (*cvFindCornerSubPixPtr)(IplImage*, CvPoint2D32f*, int,
                                       CvSize, CvSize, CvTermCriteria)
cdef cvFindCornerSubPixPtr c_cvFindCornerSubPix
c_cvFindCornerSubPix = (<cvFindCornerSubPixPtr*>
                        <size_t>ctypes.addressof(cv.cvFindCornerSubPix))[0]

# cvSmooth
ctypedef void (*cvSmoothPtr)(IplImage*, IplImage*, int, int,
                             int, double, double)
cdef cvSmoothPtr c_cvSmooth
c_cvSmooth =  (<cvSmoothPtr*><size_t>ctypes.addressof(cv.cvSmooth))[0]

# cvGoodFeaturesToTrack
ctypedef void (*cvGoodFeaturesToTrackPtr)(IplImage*, IplImage*, IplImage*,
                                          CvPoint2D32f*, int*, double, double,
                                          IplImage*, int, int, double)
cdef cvGoodFeaturesToTrackPtr c_cvGoodFeaturesToTrack
c_cvGoodFeaturesToTrack = (<cvGoodFeaturesToTrackPtr*><size_t>
                           ctypes.addressof(cv.cvGoodFeaturesToTrack))[0]

# cvGetRectSubPix
ctypedef void (*cvGetRectSubPixPtr)(IplImage*, IplImage*, CvPoint2D32f)
cdef cvGetRectSubPixPtr c_cvGetRectSubPix
c_cvGetRectSubPix = (<cvGetRectSubPixPtr*><size_t>
                     ctypes.addressof(cv.cvGetRectSubPix))[0]

# cvGetQuadrangleSubPix
ctypedef void (*cvGetQuadrangleSubPixPtr)(IplImage*, IplImage*, CvMat*)
cdef cvGetQuadrangleSubPixPtr c_cvGetQuadrangleSubPix
c_cvGetQuadrangleSubPix = (<cvGetQuadrangleSubPixPtr*><size_t>
                           ctypes.addressof(cv.cvGetQuadrangleSubPix))[0]
                           
# cvResize
ctypedef void (*cvResizePtr)(IplImage*, IplImage*, int)
cdef cvResizePtr c_cvResize
c_cvResize = (<cvResizePtr*><size_t>ctypes.addressof(cv.cvResize))[0]

# cvWarpAffine
ctypedef void (*cvWarpAffinePtr)(IplImage*, IplImage*, CvMat*, int, CvScalar)
cdef cvWarpAffinePtr c_cvWarpAffine
c_cvWarpAffine = (<cvWarpAffinePtr*><size_t>
                  ctypes.addressof(cv.cvWarpAffine))[0]

# cvWarpPerspective
ctypedef void (*cvWarpPerspectivePtr)(IplImage*, IplImage*, CvMat*, int,
                                      CvScalar)
cdef cvWarpPerspectivePtr c_cvWarpPerspective
c_cvWarpPerspective = (<cvWarpPerspectivePtr*><size_t>
                       ctypes.addressof(cv.cvWarpPerspective))[0]
                       
# cvFindChessboardCorners
ctypedef void (*cvFindChessboardCornersPtr)(IplImage*, CvSize, CvPoint2D32f*, 
                                            int*, int)
cdef cvFindChessboardCornersPtr c_cvFindChessboardCorners
c_cvFindChessboardCorners = (<cvFindChessboardCornersPtr*><size_t>
                             ctypes.addressof(cv.cvFindChessboardCorners))[0]

# cvDrawChessboardCorners
ctypedef void (*cvDrawChessboardCornersPtr)(IplImage*, CvSize, CvPoint2D32f*, 
                                            int, int)
cdef cvDrawChessboardCornersPtr c_cvDrawChessboardCorners
c_cvDrawChessboardCorners = (<cvDrawChessboardCornersPtr*><size_t>
                             ctypes.addressof(cv.cvDrawChessboardCorners))[0]


####################################
# Function Implementations
####################################


def cvSobel(np.ndarray src, np.ndarray out=None, int xorder=1, int yorder=0,
            int aperture_size=3):

    """
    better doc string needed.
    for now:
    http://opencv.willowgarage.com/documentation/cvreference.html
    """

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

    """
    better doc string needed.
    for now:
    http://opencv.willowgarage.com/documentation/cvreference.html
    """

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

    """
    better doc string needed.
    for now:
    http://opencv.willowgarage.com/documentation/cvreference.html
    """

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

def cvPreCornerDetect(np.ndarray src, np.ndarray out=None,
                      int aperture_size=3):
    """
    better doc string needed.
    for now:
    http://opencv.willowgarage.com/documentation/cvreference.html
    """

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

    """
    better doc string needed.
    for now:
    http://opencv.willowgarage.com/documentation/cvreference.html
    """

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
    """
    better doc string needed.
    for now:
    http://opencv.willowgarage.com/documentation/cvreference.html
    """
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
    """
    better doc string needed.
    for now:
    http://opencv.willowgarage.com/documentation/cvreference.html
    """
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

def cvFindCornerSubPix(np.ndarray src, np.ndarray corners, int count, win,
                       zero_zone=(-1, -1), int iterations=0,
                       double epsilon=1e-5):

    """
    better doc string needed.
    for now:
    http://opencv.willowgarage.com/documentation/cvreference.html
    """

    validate_array(src)
    validate_array(corners)

    assert_nchannels(src, [1])
    assert_dtype(src, [UINT8])

    assert_nchannels(corners, [1])
    assert_dtype(corners, [FLOAT32])

    # make sure the number of points
    # jives with the elements in the array
    # the shape of the array is irrelevant
    # because opencv will index it as if it were
    # flat anyway, but regardless, the validate_array function ensures
    # that it is 2D
    cdef int nbytes = <int> get_array_nbytes(corners)
    if nbytes != (count * 2 * 4):
        raise ValueError('The number of declared points is different '
                         'than exists in the array.')
    cdef CvPoint2D32f* cvcorners = array_as_cvPoint2D32f_ptr(corners)

    cdef CvSize cvwin
    cvwin.height = <int> win[0]
    cvwin.width = <int> win[1]

    cdef CvSize cvzerozone
    cvzerozone.height = <int> zero_zone[0]
    cvzerozone.width = <int> zero_zone[1]

    cdef IplImage srcimg
    populate_iplimage(src, &srcimg)

    cdef CvTermCriteria crit
    crit = get_cvTermCriteria(iterations, epsilon)

    c_cvFindCornerSubPix(&srcimg, cvcorners, count, cvwin, cvzerozone, crit)

    return None

def cvSmooth(np.ndarray src, np.ndarray out=None,
             int smoothtype=CV_GAUSSIAN, int param1=3,
             int param2=0, double param3=0, double param4=0,
             bool in_place=False):
    """
    better doc string needed.
    for now:
    http://opencv.willowgarage.com/documentation/cvreference.html
    """

    validate_array(src)
    if out is not None:
        validate_array(out)

    # there are restrictions that must be placed on the data depending on
    # the smoothing operation requested

    # CV_BLUR_NO_SCALE
    if smoothtype == CV_BLUR_NO_SCALE:

        if in_place:
            raise RuntimeError('In place operation not supported with this '
                               'filter')

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
            raise RuntimeError('In place operation not supported with this '
                               'filter')

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

def cvGoodFeaturesToTrack(np.ndarray src, int corner_count,
                          double quality_level, double min_distance,
                          np.ndarray mask=None, int block_size=3,
                          int use_harris=0, double k=0.04):
    """
    better doc string needed.
    for now:
    http://opencv.willowgarage.com/documentation/cvreference.html
    """

    validate_array(src)
    assert_dtype(src, [UINT8, FLOAT32])
    assert_nchannels(src, [1])

    cdef np.ndarray eig = new_array_like_diff_dtype(src, FLOAT32)
    cdef np.ndarray temp = new_array_like(eig)

    cdef np.npy_intp cornershape[2]
    cornershape[0] = <np.npy_intp>corner_count
    cornershape[1] = 2
    
    cdef np.ndarray out = new_array(2, cornershape, FLOAT32)
    cdef CvPoint2D32f* cvcorners = array_as_cvPoint2D32f_ptr(out)
    
    cdef int ncorners_found
    ncorners_found = corner_count

    cdef IplImage srcimg
    cdef IplImage eigimg
    cdef IplImage tempimg
    cdef IplImage *maskimg

    populate_iplimage(src, &srcimg)
    populate_iplimage(eig, &eigimg)
    populate_iplimage(temp, &tempimg)
    if mask is None:
        maskimg = NULL
    else:
        validate_array(mask)
        assert_nchannels(mask, [1])
        populate_iplimage(mask, maskimg)

    c_cvGoodFeaturesToTrack(&srcimg, &eigimg, &tempimg, cvcorners,
                            &ncorners_found, quality_level, min_distance,
                            maskimg, block_size,
                            use_harris, k)
    
    return out[:ncorners_found] 

def cvGetRectSubPix(np.ndarray src, size, center):
    ''' Retrieves the pixel rectangle from an image with 
    sub-pixel accuracy.
        
    Paramters:
        src - source image. 
        size - two tuple (height, width) of rectangle (ints)
        center - two tuple (x, y) of rectangle center (floats)
        
        the center must lie within the image, but the rectangle
        may extend beyond the bounds of the image, at which point
        the border is replicated.
        
    Returns:
        A new image of the extracted rectangle. The same dtype as the src image.
    '''
    
    validate_array(src)
    
    cdef np.npy_intp* shape = clone_array_shape(src)
    shape[0] = <np.npy_intp>size[0]
    shape[1] = <np.npy_intp>size[1]
    
    cdef CvPoint2D32f cvcenter
    cvcenter.x = <float>center[0]
    cvcenter.y = <float>center[1]
    
    cdef np.ndarray out = new_array(src.ndim, shape, src.dtype)
    
    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)
    
    c_cvGetRectSubPix(&srcimg, &outimg, cvcenter)
    
    PyMem_Free(shape)
    
    return out

def cvGetQuadrangleSubPix(np.ndarray src, np.ndarray warpmat, float_out=False):
    ''' Retrieves the pixel quandrangle from an image with 
    sub-pixel accuracy. In english: apply and affine transform to an image. 
    
    Parameters:
                src - input image
                warpmat - a 2x3 array which is an affine transform
                float_out - return a float32 array. If true, input must be 
                            uint8. If false, output is same type as input.
                            
    Return:
                warped image of same size and dtype as src. Except when 
                float_out == True (see above)
    '''
    validate_array(src)
    validate_array(warpmat)
    
    assert_nchannels(src, [1, 3])
    
    assert_nchannels(warpmat, [1])
    
    assert warpmat.shape[0] == 2, 'warpmat must be 2x3'
    assert warpmat.shape[1] == 3, 'warpmat must be 2x3'
    
    cdef np.ndarray out
    
    if float_out:
        assert_dtype(src, [UINT8])
        out = new_array_like_diff_dtype(src, FLOAT32)
    else:
        out = new_array_like(src)
        
    cdef IplImage srcimg
    cdef IplImage outimg
    cdef IplImage cvmat
    cdef CvMat* cvmatptr
    
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)
    populate_iplimage(warpmat, &cvmat)
    cvmatptr = cvmat_ptr_from_iplimage(&cvmat)
    
    c_cvGetQuadrangleSubPix(&srcimg, &outimg, cvmatptr)
    
    PyMem_Free(cvmatptr)
    
    return out
    
def cvResize(np.ndarray src, height=None, width=None,
             int method=CV_INTER_LINEAR):
    """
    better doc string needed.
    for now:
    http://opencv.willowgarage.com/documentation/cvreference.html
    """
    validate_array(src)

    if not height or not width:
        raise ValueError('width and height must not be none')

    cdef int ndim = src.ndim
    cdef np.npy_intp* shape = clone_array_shape(src)
    shape[0] = height
    shape[1] = width

    cdef np.ndarray out = new_array(ndim, shape, src.dtype)
    validate_array(out)

    PyMem_Free(shape)

    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)

    c_cvResize(&srcimg, &outimg, method)

    return out   
    
def cvWarpAffine(np.ndarray src, np.ndarray warpmat, 
                 int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,
                 fillval=(0., 0., 0., 0.)):
    
    ''' Applies an affine transformation to an image.
    
    Parameters:
                src - source image
                warpmat - 2x3 affine transformation
                flags - a combination of interpolation and method flags.
                        see opencv documentation for more details
                fillval - a 4 tuple of a color to fill the background
                          defaults to black. 
                          
    Returns:
                a warped image the same size and dtype as src
    '''
    validate_array(src)
    validate_array(warpmat)    
    assert len(fillval) == 4, 'fillval must be a 4-tuple'    
    assert_nchannels(src, [1, 3])    
    assert_nchannels(warpmat, [1])    
    assert warpmat.shape[0] == 2, 'warpmat must be 2x3'
    assert warpmat.shape[1] == 3, 'warpmat must be 2x3'
    
    cdef np.ndarray out
    out = new_array_like(src)
    
    cdef CvScalar cvfill
    cdef int i
    for i in range(4):
        cvfill.val[i] = <double>fillval[i]
        
    cdef IplImage srcimg
    cdef IplImage outimg
    cdef IplImage cvmat
    cdef CvMat* cvmatptr
    
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)
    populate_iplimage(warpmat, &cvmat)
    cvmatptr = cvmat_ptr_from_iplimage(&cvmat)
    
    c_cvWarpAffine(&srcimg, &outimg, cvmatptr, flags, cvfill)
    
    PyMem_Free(cvmatptr)
    
    return out
    
def cvWarpPerspective(np.ndarray src, np.ndarray warpmat, 
                      int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,
                      fillval=(0., 0., 0., 0.)):
    
    ''' Applies a perspective transformation to an image.
    
    Parameters:
                src - source image
                warpmat - 3x3 perspective transformation
                flags - a combination of interpolation and method flags.
                        see opencv documentation for more details
                fillval - a 4 tuple of a color to fill the background
                          defaults to black. 
                          
    Returns:
                a warped image the same size and dtype as src
    '''
    validate_array(src)
    validate_array(warpmat)    
    assert len(fillval) == 4, 'fillval must be a 4-tuple'    
    assert_nchannels(src, [1, 3])    
    assert_nchannels(warpmat, [1])    
    assert warpmat.shape[0] == 3, 'warpmat must be 3x3'
    assert warpmat.shape[1] == 3, 'warpmat must be 3x3'
    
    cdef np.ndarray out
    out = new_array_like(src)
    
    cdef CvScalar cvfill
    cdef int i
    for i in range(4):
        cvfill.val[i] = <double>fillval[i]
        
    cdef IplImage srcimg
    cdef IplImage outimg
    cdef IplImage cvmat
    cdef CvMat* cvmatptr
    
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)
    populate_iplimage(warpmat, &cvmat)
    cvmatptr = cvmat_ptr_from_iplimage(&cvmat)
    
    c_cvWarpPerspective(&srcimg, &outimg, cvmatptr, flags, cvfill)
    
    PyMem_Free(cvmatptr)
    
    return out                 
    
def cvFindChessboardCorners(np.ndarray src, pattern_size, 
                            int flags = CV_CALIB_CB_ADAPTIVE_THRESH):
    """
    Wrapper around the OpenCV cvFindChessboardCorners function.

    src - Image to search for chessboard corners
    pattern_size - Tuple of inner corners (h,w)
    flags - see appropriate flags in opencv docs
    http://opencv.willowgarage.com/documentation/cvreference.html
    
    returns - an nx2 array of the corners found.
    
    """
    
    validate_array(src)

    assert_nchannels(src, [1, 3])
    assert_dtype(src, [UINT8])

    cdef np.npy_intp outshape[2]
    outshape[0] = <np.npy_intp> pattern_size[0] * pattern_size[1]
    outshape[1] = <np.npy_intp> 2 

    out = new_array(2, outshape, FLOAT32)
    cdef CvPoint2D32f* cvpoints = array_as_cvPoint2D32f_ptr(out)

    cdef CvSize cvpattern_size
    cvpattern_size.height = pattern_size[0]
    cvpattern_size.width = pattern_size[1]

    cdef IplImage srcimg
    populate_iplimage(src, &srcimg)

    cdef int ncorners_found
    c_cvFindChessboardCorners(&srcimg, cvpattern_size, cvpoints, 
                              &ncorners_found, flags)
    
    return out[:ncorners_found]
    
def cvDrawChessboardCorners(np.ndarray src, pattern_size, np.ndarray corners,
                            in_place=True):
    """
    Wrapper around the OpenCV cvDrawChessboardCorners function.

    Parameters
    ----------
    src : ndarray, dim 3, dtype: uint8
        Image to draw into.
    pattern_size : array_like, shape (2,)
        Number of inner corners (w,h)
    corners : ndarray, shape (n,2), dtype: float32
        Corners found in the image. See cvFindChessboardCorners and
        cvFindCornerSubPix
    in_place: True/False (default=True) perform the drawing on the submitted
              image. If false, a copy of the image will be made and drawn to. 
    """
    validate_array(src)

    assert_nchannels(src, [3])
    assert_dtype(src, [UINT8])
    
    assert_ndims(corners, [2])
    assert_dtype(corners, [FLOAT32])
    
    cdef np.ndarray out
    
    if not in_place:
        out = src.copy()
    else:
        out = src
        
    cdef CvSize cvpattern_size
    cvpattern_size.height = pattern_size[0]
    cvpattern_size.width = pattern_size[1]

    cdef IplImage outimg
    populate_iplimage(out, &outimg)

    cdef CvPoint2D32f* cvcorners = array_as_cvPoint2D32f_ptr(corners)

    cdef int ncount = pattern_size[0] * pattern_size[1]
    
    cdef int pattern_was_found
    
    if corners.shape[0] == ncount:
        pattern_was_found = 1
    else:
        pattern_was_found = 0
    
    c_cvDrawChessboardCorners(&outimg, cvpattern_size, cvcorners,
        ncount, pattern_was_found)
        
    return out
        
        
