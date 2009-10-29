import ctypes
import numpy as np

cimport numpy as np
from python cimport *
from stdlib cimport *
from opencv_type cimport *
from opencv_backend import *
from opencv_backend cimport *
from opencv_constants import *

from opencv_constants import *
from opencv_cv import *

from _libimport import cv

# setup numpy tables for this module
np.import_array()

#-------------------------------------------------------------------------------
# Useful global stuff
#-------------------------------------------------------------------------------

# a dict for cvCvtColor to get the appropriate types and shapes without
# if statements all over the place (this way is faster, cause the dict is
# created at import time)
# the order of list arguments is:
# [in_channels, out_channels, [input_dtypes]]
# out type is always the same as in type

_cvtcolor_dict = {CV_BGR2BGRA: [3, 4, [UINT8, UINT16, FLOAT32]],
                  CV_RGB2RGBA: [3, 4, [UINT8, UINT16, FLOAT32]],
                  CV_BGRA2BGR: [4, 3, [UINT8, UINT16, FLOAT32]],
                  CV_RGBA2RGB: [4, 3, [UINT8, UINT16, FLOAT32]],
                  CV_BGR2RGBA: [3, 4, [UINT8, UINT16, FLOAT32]],
                  CV_RGB2BGRA: [3, 4, [UINT8, UINT16, FLOAT32]],
                  CV_RGBA2BGR: [4, 3, [UINT8, UINT16, FLOAT32]],
                  CV_BGRA2RGB: [4, 3, [UINT8, UINT16, FLOAT32]],
                  CV_BGR2RGB: [3, 3, [UINT8, UINT16, FLOAT32]],
                  CV_RGB2BGR: [3, 3, [UINT8, UINT16, FLOAT32]],
                  CV_BGRA2RGBA: [4, 4, [UINT8, UINT16, FLOAT32]],
                  CV_RGBA2BGRA: [4, 4, [UINT8, UINT16, FLOAT32]],
                  CV_BGR2GRAY: [3, 1, [UINT8, UINT16, FLOAT32]],
                  CV_RGB2GRAY: [3, 1, [UINT8, UINT16, FLOAT32]],
                  CV_GRAY2BGR: [1, 3, [UINT8, UINT16, FLOAT32]],
                  CV_GRAY2RGB: [1, 3, [UINT8, UINT16, FLOAT32]],
                  CV_GRAY2BGRA: [1, 4, [UINT8, UINT16, FLOAT32]],
                  CV_GRAY2RGBA: [1, 4, [UINT8, UINT16, FLOAT32]],
                  CV_BGRA2GRAY: [4, 1, [UINT8, UINT16, FLOAT32]],
                  CV_RGBA2GRAY: [4, 1, [UINT8, UINT16, FLOAT32]],
                  CV_BGR2BGR565: [3, 2, [UINT8]],
                  CV_RGB2BGR565: [3, 2, [UINT8]],
                  CV_BGR5652BGR: [2, 3, [UINT8]],
                  CV_BGR5652RGB: [2, 3, [UINT8]],
                  CV_BGRA2BGR565: [4, 2, [UINT8]],
                  CV_RGBA2BGR565: [4, 2, [UINT8]],
                  CV_BGR5652BGRA: [2, 4, [UINT8]],
                  CV_BGR5652RGBA: [2, 4, [UINT8]],
                  CV_GRAY2BGR565: [1, 2, [UINT8]],
                  CV_BGR5652GRAY: [2, 1, [UINT8]],
                  CV_BGR2BGR555: [3, 2, [UINT8]],
                  CV_RGB2BGR555: [3, 2, [UINT8]],
                  CV_BGR5552BGR: [2, 3, [UINT8]],
                  CV_BGR5552RGB: [2, 3, [UINT8]],
                  CV_BGRA2BGR555: [4, 2, [UINT8]],
                  CV_RGBA2BGR555: [4, 2, [UINT8]],
                  CV_BGR5552BGRA: [2, 4, [UINT8]],
                  CV_BGR5552RGBA: [2, 4, [UINT8]],
                  CV_GRAY2BGR555: [1, 2, [UINT8]],
                  CV_BGR5552GRAY: [2, 1, [UINT8]],
                  CV_BGR2XYZ: [3, 3, [UINT8, UINT16, FLOAT32]],
                  CV_RGB2XYZ: [3, 3, [UINT8, UINT16, FLOAT32]],
                  CV_XYZ2BGR: [3, 3, [UINT8, UINT16, FLOAT32]],
                  CV_XYZ2RGB: [3, 3, [UINT8, UINT16, FLOAT32]],
                  CV_BGR2YCrCb: [3, 3, [UINT8, UINT16, FLOAT32]],
                  CV_RGB2YCrCb: [3, 3, [UINT8, UINT16, FLOAT32]],
                  CV_YCrCb2BGR: [3, 3, [UINT8, UINT16, FLOAT32]],
                  CV_YCrCb2RGB: [3, 3, [UINT8, UINT16, FLOAT32]],
                  CV_BGR2HSV: [3, 3, [UINT8, FLOAT32]],
                  CV_RGB2HSV: [3, 3, [UINT8, FLOAT32]],
                  CV_BGR2Lab: [3, 3, [UINT8, FLOAT32]],
                  CV_RGB2Lab: [3, 3, [UINT8, FLOAT32]],
                  CV_BayerBG2BGR: [1, 3, [UINT8]],
                  CV_BayerGB2BGR: [1, 3, [UINT8]],
                  CV_BayerRG2BGR: [1, 3, [UINT8]],
                  CV_BayerGR2BGR: [1, 3, [UINT8]],
                  CV_BayerBG2RGB: [1, 3, [UINT8]],
                  CV_BayerGB2RGB: [1, 3, [UINT8]],
                  CV_BayerRG2RGB: [1, 3, [UINT8]],
                  CV_BayerGR2RGB: [1, 3, [UINT8]],
                  CV_BGR2Luv: [3, 3, [UINT8, FLOAT32]],
                  CV_RGB2Luv: [3, 3, [UINT8, FLOAT32]],
                  CV_BGR2HLS: [3, 3, [UINT8, FLOAT32]],
                  CV_RGB2HLS: [3, 3, [UINT8, FLOAT32]],
                  CV_HSV2BGR: [3, 3, [UINT8, FLOAT32]],
                  CV_HSV2RGB: [3, 3, [UINT8, FLOAT32]],
                  CV_Lab2BGR: [3, 3, [UINT8, FLOAT32]],
                  CV_Lab2RGB: [3, 3, [UINT8, FLOAT32]],
                  CV_Luv2BGR: [3, 3, [UINT8, FLOAT32]],
                  CV_Luv2RGB: [3, 3, [UINT8, FLOAT32]],
                  CV_HLS2BGR: [3, 3, [UINT8, FLOAT32]],
                  CV_HLS2RGB: [3, 3, [UINT8, FLOAT32]]}


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

# cvLogPolar
ctypedef void (*cvLogPolarPtr)(IplImage*, IplImage*, CvPoint2D32f, double, int)
cdef cvLogPolarPtr c_cvLogPolar
c_cvLogPolar = (<cvLogPolarPtr*><size_t>ctypes.addressof(cv.cvLogPolar))[0]

# cvErode
ctypedef void (*cvErodePtr)(IplImage*, IplImage*, IplConvKernel*, int)
cdef cvErodePtr c_cvErode
c_cvErode = (<cvErodePtr*><size_t>ctypes.addressof(cv.cvErode))[0]

# cvDilate
ctypedef void (*cvDilatePtr)(IplImage*, IplImage*, IplConvKernel*, int)
cdef cvDilatePtr c_cvDilate
c_cvDilate = (<cvDilatePtr*><size_t>ctypes.addressof(cv.cvDilate))[0]

# cvMorphologyEx
ctypedef void (*cvMorphologyExPtr)(IplImage*, IplImage*, IplImage*,
                                   IplConvKernel*, int, int)
cdef cvMorphologyExPtr c_cvMorphologyEx
c_cvMorphologyEx = (<cvMorphologyExPtr*><size_t>
                        ctypes.addressof(cv.cvMorphologyEx))[0]

# cvSmooth
ctypedef void (*cvSmoothPtr)(IplImage*, IplImage*, int, int,
                             int, double, double)
cdef cvSmoothPtr c_cvSmooth
c_cvSmooth =  (<cvSmoothPtr*><size_t>ctypes.addressof(cv.cvSmooth))[0]

# cvFilter2D
ctypedef void (*cvFilter2DPtr)(IplImage*, IplImage*, CvMat*, CvPoint)
cdef cvFilter2DPtr c_cvFilter2D
c_cvFilter2D = (<cvFilter2DPtr*><size_t>ctypes.addressof(cv.cvFilter2D))[0]

# cvIntegral
ctypedef void (*cvIntegralPtr)(IplImage*, IplImage*, IplImage*, IplImage*)
cdef cvIntegralPtr c_cvIntegral
c_cvIntegral = (<cvIntegralPtr*><size_t>ctypes.addressof(cv.cvIntegral))[0]

# cvCvtColor
ctypedef void (*cvCvtColorPtr)(IplImage*, IplImage*, int)
cdef cvCvtColorPtr c_cvCvtColor
c_cvCvtColor = (<cvCvtColorPtr*><size_t>ctypes.addressof(cv.cvCvtColor))[0]

# cvThreshold
ctypedef double (*cvThresholdPtr)(IplImage*, IplImage*, double, double, int)
cdef cvThresholdPtr c_cvThreshold
c_cvThreshold = (<cvThresholdPtr*><size_t>ctypes.addressof(cv.cvThreshold))[0]

# cvAdaptiveThreshold
ctypedef void (*cvAdaptiveThresholdPtr)(IplImage*, IplImage*, double, int, int,
                                        int, double)
cdef cvAdaptiveThresholdPtr c_cvAdaptiveThreshold
c_cvAdaptiveThreshold = (<cvAdaptiveThresholdPtr*><size_t>
                         ctypes.addressof(cv.cvAdaptiveThreshold))[0]

# cvPyrDown
ctypedef void (*cvPyrDownPtr)(IplImage*, IplImage*, int)
cdef cvPyrDownPtr c_cvPyrDown
c_cvPyrDown = (<cvPyrDownPtr*><size_t>ctypes.addressof(cv.cvPyrDown))[0]

# cvPyrUp
ctypedef void (*cvPyrUpPtr)(IplImage*, IplImage*, int)
cdef cvPyrUpPtr c_cvPyrUp
c_cvPyrUp = (<cvPyrUpPtr*><size_t>ctypes.addressof(cv.cvPyrUp))[0]

# cvCalibrateCamera2
ctypedef void (*cvCalibrateCamera2Ptr)(CvMat*, CvMat*, CvMat*,
       CvSize, CvMat*, CvMat*, CvMat*, CvMat*, int)
cdef cvCalibrateCamera2Ptr c_cvCalibrateCamera2
c_cvCalibrateCamera2 = (<cvCalibrateCamera2Ptr*>
                        <size_t>ctypes.addressof(cv.cvCalibrateCamera2))[0]

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
def cvSobel(np.ndarray src, int xorder=1, int yorder=0,
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

    cdef np.ndarray out

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

def cvLaplace(np.ndarray src, int aperture_size=3):

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

    cdef np.ndarray out

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

def cvCanny(np.ndarray src, double threshold1=10, double threshold2=50,
            int aperture_size=3):

    """
    better doc string needed.
    for now:
    http://opencv.willowgarage.com/documentation/cvreference.html
    """

    validate_array(src)
    assert_dtype(src, [UINT8])
    assert_nchannels(src, [1])

    if (aperture_size != 3 and aperture_size != 5 and aperture_size != 7):
        raise ValueError('aperture_size must be 3, 5, or 7')

    cdef np.ndarray out
    out = new_array_like(src)

    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)

    c_cvCanny(&srcimg, &outimg, threshold1, threshold2, aperture_size)

    return out

def cvPreCornerDetect(np.ndarray src, int aperture_size=3):
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

    cdef np.ndarray out
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

    validate_array(src)
    assert_nchannels(src, [1])
    assert_dtype(src, [UINT8, FLOAT32])

    if (aperture_size != 3 and aperture_size != 5 and aperture_size != 7):
        raise ValueError('aperture_size must be 3, 5, or 7')

    cdef np.ndarray out
    cdef np.npy_intp outshape[2]
    outshape[0] = src.shape[0]
    outshape[1] = src.shape[1] * 6

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

    validate_array(src)
    assert_nchannels(src, [1])
    assert_dtype(src, [UINT8, FLOAT32])

    if (aperture_size != 3 and aperture_size != 5 and aperture_size != 7):
        raise ValueError('aperture_size must be 3, 5, or 7')

    cdef np.ndarray out
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

    validate_array(src)
    assert_nchannels(src, [1])
    assert_dtype(src, [UINT8, FLOAT32])

    if (aperture_size != 3 and aperture_size != 5 and aperture_size != 7):
        raise ValueError('aperture_size must be 3, 5, or 7')

    cdef np.ndarray out
    out = new_array_like_diff_dtype(src, FLOAT32)

    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)

    c_cvCornerHarris(&srcimg, &outimg, block_size, aperture_size, k)

    return out

def cvFindCornerSubPix(np.ndarray src, np.ndarray corners, win,
                       zero_zone=(-1, -1), int iterations=0,
                       double epsilon=1e-5):

    """
    better doc string needed.
    for now:
    http://opencv.willowgarage.com/documentation/cvreference.html
    """

    validate_array(src)
    assert_nchannels(src, [1])
    assert_dtype(src, [UINT8])

    validate_array(corners)
    assert_ndims(corners, [2])
    assert_dtype(corners, [FLOAT32])

    cdef int count = <int>(corners.shape[0] * corners.shape[1] / 2.)
    cdef CvPoint2D32f* cvcorners = array_as_cvPoint2D32f_ptr(corners)

    if len(win) != 2:
        raise ValueError('win must be a 2-tuple')
    cdef CvSize cvwin
    cvwin.height = <int> win[0]
    cvwin.width = <int> win[1]

    cdef int imgheight = src.shape[0]
    cdef int imgwidth = src.shape[1]
    if imgwidth < (cvwin.width * 2 + 5) or imgheight  < (cvwin.height * 2 + 5):
        raise ValueError('The window is too large.')

    cdef CvSize cvzerozone
    cvzerozone.height = <int> zero_zone[0]
    cvzerozone.width = <int> zero_zone[1]

    cdef IplImage srcimg
    populate_iplimage(src, &srcimg)

    cdef CvTermCriteria crit
    crit = get_cvTermCriteria(iterations, epsilon)

    c_cvFindCornerSubPix(&srcimg, cvcorners, count, cvwin, cvzerozone, crit)

    return corners

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

    if warpmat.shape[0] != 2 or warpmat.shape[1] != 3:
        raise ValueError('warpmat must be 2x3')

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
    shape[0] = <np.npy_intp>height
    shape[1] = <np.npy_intp>width

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
    if len(fillval) != 4:
        raise ValueError('fillval must be a 4-tuple')
    assert_nchannels(src, [1, 3])
    assert_nchannels(warpmat, [1])

    if warpmat.shape[0] != 2 or warpmat.shape[1] != 3:
        raise ValueError('warpmat must be 2x3')

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
    if len(fillval) != 4:
        raise ValueError('fillval must be a 4-tuple')
    assert_nchannels(src, [1, 3])
    assert_nchannels(warpmat, [1])
    if warpmat.shape[0] != 3 or warpmat.shape[1] != 3:
        raise ValueError('warpmat must be 3x3')

    cdef np.ndarray out
    out = new_array_like(src)

    cdef CvScalar cvfill
    cdef int i
    for i in range(4):
        cvfill.val[i] = <double>fillval[i]

    cdef IplImage srcimg
    cdef IplImage outimg
    cdef IplImage cvmat
    cdef CvMat* cvmatptr = NULL

    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)
    populate_iplimage(warpmat, &cvmat)
    cvmatptr = cvmat_ptr_from_iplimage(&cvmat)
    c_cvWarpPerspective(&srcimg, &outimg, cvmatptr, flags, cvfill)

    PyMem_Free(cvmatptr)

    return out

def cvLogPolar(np.ndarray src, center, double M,
               int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS):

    validate_array(src)
    if len(center) != 2:
        raise ValueError('center must be a 2-tuple')

    cdef np.ndarray out = new_array_like(src)

    cdef CvPoint2D32f cv_center
    cv_center.x = <float>center[0]
    cv_center.y = <float>center[1]

    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)

    c_cvLogPolar(&srcimg, &outimg, cv_center, M, flags)
    return out

def cvErode(np.ndarray src, np.ndarray element=None, int iterations=1,
            anchor=None, in_place=False):

    validate_array(src)

    cdef np.ndarray out
    cdef IplConvKernel* iplkernel

    if element == None:
        iplkernel = NULL
    else:
        iplkernel = get_IplConvKernel_ptr_from_array(element, anchor)

    if in_place:
        out = src
    else:
        out = new_array_like(src)

    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)

    c_cvErode(&srcimg, &outimg, iplkernel, iterations)

    free_IplConvKernel(iplkernel)

    if in_place:
        return None
    else:
        return out

def cvDilate(np.ndarray src, np.ndarray element=None, int iterations=1,
            anchor=None, in_place=False):

    validate_array(src)

    cdef np.ndarray out
    cdef IplConvKernel* iplkernel

    if element == None:
        iplkernel = NULL
    else:
        iplkernel = get_IplConvKernel_ptr_from_array(element, anchor)

    if in_place:
        out = src
    else:
        out = new_array_like(src)

    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)

    c_cvDilate(&srcimg, &outimg, iplkernel, iterations)

    free_IplConvKernel(iplkernel)

    if in_place:
        return None
    else:
        return out

def cvMorphologyEx(np.ndarray src, np.ndarray element, int operation,
                   int iterations=1, anchor=None, in_place=False):

    validate_array(src)

    cdef np.ndarray out
    cdef np.ndarray temp
    cdef IplConvKernel* iplkernel

    iplkernel = get_IplConvKernel_ptr_from_array(element, anchor)

    if in_place:
        out = src
    else:
        out = new_array_like(src)

    cdef IplImage srcimg
    cdef IplImage outimg
    cdef IplImage tempimg
    cdef IplImage* tempimgptr = &tempimg

    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)

    # determine if we need the tempimg
    if operation == CV_MOP_OPEN or operation == CV_MOP_CLOSE:
        tempimgptr = NULL
    elif operation == CV_MOP_GRADIENT:
        temp = new_array_like(src)
        populate_iplimage(temp, &tempimg)
    elif operation == CV_MOP_TOPHAT or operation == CV_MOP_BLACKHAT:
        if in_place:
            temp = new_array_like(src)
            populate_iplimage(temp, &tempimg)
        else:
            tempimgptr = NULL
    else:
        raise RuntimeError('operation type not understood')

    c_cvMorphologyEx(&srcimg, &outimg, tempimgptr, iplkernel, operation,
                     iterations)

    free_IplConvKernel(iplkernel)

    if in_place:
        return None
    else:
        return out

def cvSmooth(np.ndarray src, int smoothtype=CV_GAUSSIAN, int param1=3,
             int param2=0, double param3=0, double param4=0,
             bool in_place=False):
    """
    better doc string needed.
    for now:
    http://opencv.willowgarage.com/documentation/cvreference.html
    """

    validate_array(src)

    cdef np.ndarray out
    # there are restrictions that must be placed on the data depending on
    # the smoothing operation requested

    # CV_BLUR_NO_SCALE
    if smoothtype == CV_BLUR_NO_SCALE:

        if in_place:
            raise RuntimeError('In place operation not supported with this '
                               'filter')

        assert_dtype(src, [UINT8, INT8, FLOAT32])
        assert_ndims(src, [2])

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
        else:
            out = new_array_like(src)

    # CV_MEDIAN and CV_BILATERAL
    else:
        assert_dtype(src, [UINT8, INT8])
        assert_nchannels(src, [1, 3])

        if in_place:
            raise RuntimeError('In place operation not supported with this '
                               'filter')

        out = new_array_like(src)

    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)

    c_cvSmooth(&srcimg, &outimg, smoothtype, param1, param2, param3, param4)

    if in_place:
        return None
    else:
        return out

def cvFilter2D(np.ndarray src, np.ndarray kernel, anchor=None, in_place=False):

    validate_array(src)
    validate_array(kernel)

    assert_ndims(kernel, [2])
    assert_dtype(kernel, [FLOAT32])

    cdef CvPoint cv_anchor
    if anchor is not None:
        assert len(anchor) == 2, 'anchor must be (x, y) tuple'
        cv_anchor.x = <int>anchor[0]
        cv_anchor.y = <int>anchor[1]
        assert (cv_anchor.x < kernel.shape[1]) and (cv_anchor.x >= 0) \
            and (cv_anchor.y < kernel.shape[0]) and (cv_anchor.y >= 0), \
            'anchor point must be inside kernel'
    else:
        cv_anchor.x = <int>(kernel.shape[1] / 2.)
        cv_anchor.y = <int>(kernel.shape[0] / 2.)

    cdef np.ndarray out

    if in_place:
        out = src
    else:
        out = new_array_like(src)

    cdef IplImage srcimg
    cdef IplImage outimg
    cdef IplImage kernelimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)
    populate_iplimage(kernel, &kernelimg)

    cdef CvMat* cv_kernel
    cv_kernel = cvmat_ptr_from_iplimage(&kernelimg)

    c_cvFilter2D(&srcimg, &outimg, cv_kernel, cv_anchor)

    PyMem_Free(cv_kernel)

    if in_place:
        return None
    else:
        return out

def cvIntegral(np.ndarray src, square_sum=False, tilted_sum=False):

    validate_array(src)
    assert_dtype(src, [UINT8, FLOAT32, FLOAT64])

    out = []

    cdef np.ndarray outsum
    cdef np.ndarray outsqsum
    cdef np.ndarray outtiltsum

    cdef IplImage srcimg
    cdef IplImage outsumimg
    cdef IplImage outsqsumimg
    cdef IplImage outtiltsumimg
    cdef IplImage* outsqsumimgptr = &outsqsumimg
    cdef IplImage* outtiltsumimgptr = &outtiltsumimg

    populate_iplimage(src, &srcimg)

    # out arrays need to be (H + 1) x (W + 1)
    cdef np.npy_intp* out_shape = clone_array_shape(src)
    out_shape[0] = src.shape[0] + 1
    out_shape[1] = src.shape[1] + 1
    cdef int out_dims = src.ndim

    if src.dtype == UINT8:
        outsum = new_array(out_dims, out_shape, INT32)
    else:
        outsum = new_array(out_dims, out_shape, FLOAT64)

    populate_iplimage(outsum, &outsumimg)
    out.append(outsum)

    if square_sum:
        outsqsum = new_array(out_dims, out_shape, FLOAT64)
        populate_iplimage(outsqsum, &outsqsumimg)
        out.append(outsqsum)
    else:
        outsqsumimgptr = NULL

    if tilted_sum:
        outtiltsum = new_array(out_dims, out_shape, outsum.dtype)
        populate_iplimage(outtiltsum, &outtiltsumimg)
        out.append(outtiltsum)
    else:
        outtiltsumimgptr = NULL

    c_cvIntegral(&srcimg, &outsumimg, outsqsumimgptr, outtiltsumimgptr)

    PyMem_Free(out_shape)

    return out

def cvCvtColor(np.ndarray src, int code):

    validate_array(src)
    assert_dtype(src, [UINT8, UINT16, FLOAT32])

    try:
        conversion_params = _cvtcolor_dict[code]
    except KeyError:
        print 'unknown conversion code'
        raise

    cdef int src_channels = <int>conversion_params[0]
    cdef int out_channels = <int>conversion_params[1]
    src_dtypes = conversion_params[2]

    assert_nchannels(src, src_channels)
    assert_dtype(src, src_dtypes)

    cdef np.ndarray out

    # the out array can be 2, 3, or 4 channels so we need shapes that
    # can handle either
    cdef np.npy_intp out_shape2[2]
    cdef np.npy_intp out_shape3[3]
    out_shape2[0] = src.shape[0]
    out_shape2[1] = src.shape[1]
    out_shape3[0] = src.shape[0]
    out_shape3[1] = src.shape[1]

    if out_channels == 1:
        out = new_array(2, out_shape2, src.dtype)
    else:
        out_shape3[2] = <np.npy_intp>out_channels
        out = new_array(3, out_shape3, src.dtype)

    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)

    c_cvCvtColor(&srcimg, &outimg, code)

    return out

def cvThreshold(np.ndarray src, double threshold, double max_value=255,
                int threshold_type=CV_THRESH_BINARY, use_otsu=False):

    validate_array(src)
    assert_nchannels(src, [1])
    assert_dtype(src, [UINT8, FLOAT32])

    if use_otsu:
        assert_dtype(src, [UINT8])
        threshold_type += 8

    cdef np.ndarray out = new_array_like(src)

    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)

    threshold = c_cvThreshold(&srcimg, &outimg, threshold, max_value,
                              threshold_type)

    if use_otsu:
        return (out, threshold)
    else:
        return out

def cvAdaptiveThreshold(np.ndarray src, double max_value,
                        int adaptive_method=CV_ADAPTIVE_THRESH_MEAN_C,
                        int threshold_type=CV_THRESH_BINARY,
                        int block_size=3, double param1=5):

    validate_array(src)
    assert_nchannels(src, [1])
    assert_dtype(src, [UINT8])

    if (adaptive_method!=CV_ADAPTIVE_THRESH_MEAN_C and
        adaptive_method!=CV_ADAPTIVE_THRESH_GAUSSIAN_C):
        raise ValueError('Invalid adaptive method')

    if (threshold_type!=CV_THRESH_BINARY and
        threshold_type!=CV_THRESH_BINARY_INV):
        raise ValueError('Invalid threshold type')

    if (block_size % 2 != 1 or block_size <= 1):
        raise ValueError('block size must be and odd number and greater than 1')

    cdef np.ndarray out = new_array_like(src)

    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)

    c_cvAdaptiveThreshold(&srcimg, &outimg, max_value, adaptive_method,
                          threshold_type, block_size, param1)

    return out

def cvPyrDown(np.ndarray src):

    validate_array(src)
    assert_dtype(src, [UINT8, UINT16, FLOAT32, FLOAT64])

    cdef int outdim = src.ndim
    cdef np.npy_intp* outshape = clone_array_shape(src)
    outshape[0] = <np.npy_intp>(src.shape[0] + 1) / 2
    outshape[1] = <np.npy_intp>(src.shape[1] + 1) / 2

    cdef np.ndarray out = new_array(outdim, outshape, src.dtype)

    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)

    c_cvPyrDown(&srcimg, &outimg, 7)

    PyMem_Free(outshape)

    return out

def cvPyrUp(np.ndarray src):

    validate_array(src)
    assert_dtype(src, [UINT8, UINT16, FLOAT32, FLOAT64])

    cdef int outdim = src.ndim
    cdef np.npy_intp* outshape = clone_array_shape(src)
    outshape[0] = <np.npy_intp>(src.shape[0] * 2)
    outshape[1] = <np.npy_intp>(src.shape[1] * 2)

    cdef np.ndarray out = new_array(outdim, outshape, src.dtype)

    cdef IplImage srcimg
    cdef IplImage outimg
    populate_iplimage(src, &srcimg)
    populate_iplimage(out, &outimg)

    c_cvPyrUp(&srcimg, &outimg, 7)

    PyMem_Free(outshape)

    return out

def cvCalibrateCamera2(np.ndarray object_points, np.ndarray image_points,
           np.ndarray point_counts, image_size):

    # Validate input
    validate_array(object_points)
    assert_ndims(object_points, [2])

    validate_array(image_points)
    assert_ndims(image_points, [2])

    assert_dtype(point_counts, [INT32])
    assert_ndims(point_counts, [1])

    # Allocate a new intrinsics array
    cdef np.npy_intp intrinsics_shape[2]
    intrinsics_shape[0] = <np.npy_intp> 3
    intrinsics_shape[1] = <np.npy_intp> 3
    cdef np.ndarray intrinsics = new_array(2, intrinsics_shape, FLOAT64)
    cdef IplImage ipl_intrinsics
    populate_iplimage(intrinsics, &ipl_intrinsics)
    cdef CvMat* cvmat_intrinsics = cvmat_ptr_from_iplimage(&ipl_intrinsics)

    # Allocate a new distortion array
    cdef np.npy_intp distortion_shape[2]
    distortion_shape[0] = <np.npy_intp> 1
    distortion_shape[1] = <np.npy_intp> 5
    cdef np.ndarray distortion = new_array(2, distortion_shape, FLOAT64)
    cdef IplImage ipl_distortion
    populate_iplimage(distortion, &ipl_distortion)
    cdef CvMat* cvmat_distortion = cvmat_ptr_from_iplimage(&ipl_distortion)

    # Make the object & image points & npoints accessible for OpenCV
    cdef IplImage ipl_object_points, ipl_image_points, ipl_point_counts
    cdef CvMat* cvmat_object_points, *cvmat_image_points, *cvmat_point_counts
    populate_iplimage(object_points, &ipl_object_points)
    populate_iplimage(image_points, &ipl_image_points)
    populate_iplimage(point_counts, &ipl_point_counts)

    cvmat_object_points = cvmat_ptr_from_iplimage(&ipl_object_points)
    cvmat_image_points = cvmat_ptr_from_iplimage(&ipl_image_points)
    cvmat_point_counts = cvmat_ptr_from_iplimage(&ipl_point_counts)

    # Set image size
    cdef CvSize cv_image_size
    cv_image_size.height = image_size[0]
    cv_image_size.width = image_size[1]

    # Call the function
    c_cvCalibrateCamera2(cvmat_object_points, cvmat_image_points,
                         cvmat_point_counts, cv_image_size, cvmat_intrinsics,
                         cvmat_distortion, NULL, NULL, 0)

    # Convert distortion back into a vector
    distortion = np.PyArray_Squeeze(distortion)

    PyMem_Free(cvmat_intrinsics)
    PyMem_Free(cvmat_distortion)
    PyMem_Free(cvmat_object_points)
    PyMem_Free(cvmat_image_points)
    PyMem_Free(cvmat_point_counts)

    return intrinsics, distortion

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

    cdef np.ndarray out
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
                            in_place=False):
    """
    Wrapper around the OpenCV cvDrawChessboardCorners function.

    Parameters
    ----------
    src : ndarray, dim 3, dtype: uint8
        Image to draw into.
    pattern_size : array_like, shape (2,)
        Number of inner corners (h,w)
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

    if in_place:
        return None
    else:
        return out


