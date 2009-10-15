import ctypes
import numpy as np
cimport numpy as np
from python cimport *
#from stdlib cimport *
from opencv_type cimport *
from opencv_backend import *
from opencv_backend cimport *
from opencv_constants import *

from _libimport import cv

from opencv_constants import *
from opencv_cv import *


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


# cvFindCornerSubPix
ctypedef void (*cvFindCornerSubPixPtr)(IplImage*, CvPoint2D32f*, int, CvSize, CvSize, CvTermCriteria)
cdef cvFindCornerSubPixPtr c_cvFindCornerSubPix
c_cvFindCornerSubPix = (<cvFindCornerSubPixPtr*><size_t>ctypes.addressof(cv.cvFindCornerSubPix))[0]

# cvFindChessboardCorners
ctypedef void (*cvFindChessboardCornersPtr)(IplImage*, CvSize, CvPoint2D32f*, int*, int)
cdef cvFindChessboardCornersPtr c_cvFindChessboardCorners
c_cvFindChessboardCorners = (<cvFindChessboardCornersPtr*><size_t>ctypes.addressof(cv.cvFindChessboardCorners))[0]

# cvDrawChessboardCorners
ctypedef void (*cvDrawChessboardCornersPtr)(IplImage*, CvSize, CvPoint2D32f*, int, int)
cdef cvDrawChessboardCornersPtr c_cvDrawChessboardCorners
c_cvDrawChessboardCorners = (<cvDrawChessboardCornersPtr*><size_t>ctypes.addressof(cv.cvDrawChessboardCorners))[0]

# cvSmooth
ctypedef void (*cvSmoothPtr)(IplImage*, IplImage*, int, int, int, double, double)
cdef cvSmoothPtr c_cvSmooth
c_cvSmooth =  (<cvSmoothPtr*><size_t>ctypes.addressof(cv.cvSmooth))[0]

# cvGoodFeaturesToTrack
ctypedef void (*cvGoodFeaturesToTrackPtr)(IplImage*, IplImage*, IplImage*,
                                          CvPoint2D32f*, int*, double, double,
                                          IplImage*, int, int, double)
cdef cvGoodFeaturesToTrackPtr c_cvGoodFeaturesToTrack
c_cvGoodFeaturesToTrack = (<cvGoodFeaturesToTrackPtr*><size_t>ctypes.addressof(cv.cvGoodFeaturesToTrack))[0]

# cvResize
ctypedef void (*cvResizePtr)(IplImage*, IplImage*, int)
cdef cvResizePtr c_cvResize
c_cvResize = (<cvResizePtr*><size_t>ctypes.addressof(cv.cvResize))[0]


####################################
# Function Implementations
####################################
def cvFindChessboardCorners(np.ndarray src, pattern_size, int flags = CV_CALIB_CB_ADAPTIVE_THRESH):
    """
    Wrapper around the OpenCV cvFindChessboardCorners function.

    src - Image to search for chessboard corners
    pattern_size - Tuple of inner corners (w,h)
    flags - directly passed through to OpenCV
    """
    validate_array(src)

    assert_nchannels(src, [1, 3])
    assert_dtype(src, [UINT8])

    cdef np.npy_intp outshape[2]
    outshape[0] = <int> pattern_size[1]*pattern_size[0]
    outshape[1] = <int> 2 # pattern_size[0]

    points = new_array(2, outshape, FLOAT32)
    cdef CvPoint2D32f* cvpoints = array_as_cvPoint2D32f_ptr(points)

    cdef CvSize cvpattern_size
    cvpattern_size.height = pattern_size[1]
    cvpattern_size.width = pattern_size[0]

    cdef IplImage srcimg
    populate_iplimage(src, &srcimg)

    cdef int ncorners_found
    c_cvFindChessboardCorners(&srcimg, cvpattern_size, cvpoints, &ncorners_found, flags)

    return points[:ncorners_found]

def cvDrawChessboardCorners(np.ndarray out, pattern_size, np.ndarray corners):
    """
    Wrapper around the OpenCV cvDrawChessboardCorners function.

    Parameters
    ----------
    out : ndarray, dim 3, dtype: uint8
        Image to draw into
    pattern_size : array_like, shape (2,)
        Number of inner corners (w,h)
    corners : ndarray, shape (n,2), dtype: float32
        Corners found in the image. See cvFindChessboardCorners and
        cvFindCornerSubPix
    """
    validate_array(out)

    assert_nchannels(out, [3])
    assert_dtype(out, [UINT8])

    cdef CvSize cvpattern_size
    cvpattern_size.height = pattern_size[1]
    cvpattern_size.width = pattern_size[0]

    cdef IplImage img
    populate_iplimage(out, &img)

    cdef CvPoint2D32f* cvcorners = array_as_cvPoint2D32f_ptr(corners)

    cdef int ncount = pattern_size[0]*pattern_size[1]
    c_cvDrawChessboardCorners(&img, cvpattern_size, cvcorners,
        ncount, <int> len(corners) == ncount)

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

def cvPreCornerDetect(np.ndarray src, np.ndarray out=None, int aperture_size=3):

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
        raise ValueError('the number of declared points is different than exists in the array')

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

def cvSmooth(np.ndarray src, np.ndarray out=None, int smoothtype=CV_GAUSSIAN, int param1=3,
            int param2=0, double param3=0, double param4=0, bool in_place=False):
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

def cvGoodFeaturesToTrack(np.ndarray src, int corner_count, double quality_level,
                          double min_distance, np.ndarray mask=None,
                          int block_size=3, int use_harris=0, double k=0.04):
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

    cdef CvPoint2D32f* corners = (
            <CvPoint2D32f*>PyMem_Malloc(corner_count * sizeof(CvPoint2D32f)))

    cdef int out_corner_count
    out_corner_count = corner_count

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

    c_cvGoodFeaturesToTrack(&srcimg, &eigimg, &tempimg, corners, &out_corner_count,
                            quality_level, min_distance, maskimg, block_size,
                            use_harris, k)

    # since the maximum allowed corners may not have been found
    # the array might be too long, we create a new array and copy
    # the the data into it
    #
    # It would be nice to use the numpy C-Api for this, but I couldn't quite
    # get it to work

    cdef np.npy_intp cornershape[2]
    cornershape[0] = <np.npy_intp>out_corner_count
    cornershape[1] = 2

    cdef np.ndarray cornersarr = new_array(2, cornershape, FLOAT32)
    cdef int i
    for i in range(out_corner_count):
        cornersarr[i,0] = corners[i].x
        cornersarr[i,1] = corners[i].y

    PyMem_Free(corners)

    return cornersarr


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

