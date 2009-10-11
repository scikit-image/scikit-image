# test for the opencv_cv extension module

import numpy as np
from numpy.testing import *

try:
    from scikits.image.opencv import *
    OPENCV_LIBS_NOTFOUND = False
except:
    OPENCV_LIBS_NOTFOUND = True

@dec.skipif(OPENCV_LIBS_NOTFOUND, 'Skipping OpenCV test because OpenCV libs were not found')
def test_cvSobel():
    lena_RGB_U8 = np.load('data/lena_RGB_U8.npy')
    lena_GRAY_U8 = np.load('data/lena_GRAY_U8.npy')
    lena_SOBEL_INT16 = np.load('data/lena_SOBEL_INT16.npy')
    sobel = cvSobel(lena_GRAY_U8)
    assert_array_equal(sobel, lena_SOBEL_INT16)
    assert_raises(Exception, cvSobel, lena_RGB_U8)
    assert_raises(Exception, cvSobel, (lena_GRAY_U8, lena_GRAY_U8))
    assert_raises(Exception, cvSobel, lena_GRAY_U8, {'aperture_size':1})
    assert_raises(Exception, cvSobel, lena_GRAY_U8, {'aperture_size':2})
    assert_raises(Exception, cvSobel, lena_GRAY_U8, {'aperture_size':4})
    assert_raises(Exception, cvSobel, lena_GRAY_U8, {'aperture_size':6})
    assert_raises(Exception, cvSobel, lena_GRAY_U8, {'aperture_size':8})
    # any need to keep going? only valid apertures are 3, 5, and 7 
    # and testing the validity would require saving new images for each one

@dec.skipif(OPENCV_LIBS_NOTFOUND, 'Skipping OpenCV test because OpenCV libs were not found')
def test_cvLaplace():
    lena_RGB_U8 = np.load('data/lena_RGB_U8.npy')
    lena_GRAY_U8 = np.load('data/lena_GRAY_U8.npy')
    lena_LAPLACE_INT16 = np.load('data/lena_LAPLACE_INT16.npy')
    laplace = cvLaplace(lena_GRAY_U8)
    assert_array_equal(laplace, lena_LAPLACE_INT16)
    assert_raises(Exception, cvLaplace, lena_RGB_U8)
    assert_raises(Exception, cvLaplace, (lena_GRAY_U8, lena_GRAY_U8))
    assert_raises(Exception, cvLaplace, lena_GRAY_U8, {'aperture_size':1})
    assert_raises(Exception, cvLaplace, lena_GRAY_U8, {'aperture_size':2})
    assert_raises(Exception, cvLaplace, lena_GRAY_U8, {'aperture_size':4})
    assert_raises(Exception, cvLaplace, lena_GRAY_U8, {'aperture_size':6})
    assert_raises(Exception, cvLaplace, lena_GRAY_U8, {'aperture_size':8})
    # any need to keep going? only valid apertures are 3, 5, and 7 
    # and testing the validity would require saving new images for each one

@dec.skipif(OPENCV_LIBS_NOTFOUND, 'Skipping OpenCV test because OpenCV libs were not found')    
def test_cvCanny():
    lena_RGB_U8 = np.load('data/lena_RGB_U8.npy')
    lena_GRAY_U8 = np.load('data/lena_GRAY_U8.npy')
    lena_CANNY_U8 = np.load('data/lena_CANNY_U8.npy')
    canny = cvCanny(lena_GRAY_U8)
    assert_array_equal(canny, lena_CANNY_U8)
    assert_raises(Exception, cvCanny, lena_RGB_U8)
    assert_raises(Exception, cvCanny, (lena_GRAY_U8, lena_GRAY_U8))
    assert_raises(Exception, cvCanny, lena_GRAY_U8, {'aperture_size':1})
    assert_raises(Exception, cvCanny, lena_GRAY_U8, {'aperture_size':2})
    assert_raises(Exception, cvCanny, lena_GRAY_U8, {'aperture_size':4})
    assert_raises(Exception, cvCanny, lena_GRAY_U8, {'aperture_size':6})
    assert_raises(Exception, cvCanny, lena_GRAY_U8, {'aperture_size':8})
    # any need to keep going? only valid apertures are 3, 5, and 7 
    # and testing the validity would require saving new images for each one

@dec.skipif(OPENCV_LIBS_NOTFOUND, 'Skipping OpenCV test because OpenCV libs were not found')   
def test_cvPreCornerDetect():
    lena_RGB_U8 = np.load('data/lena_RGB_U8.npy')
    lena_GRAY_U8 = np.load('data/lena_GRAY_U8.npy')
    lena_GRAY_FLOAT32 = np.load('data/lena_GRAY_FLOAT32.npy')
    lena_PRECORNERDETECT_U8 = np.load('data/lena_PRECORNERDETECT_U8.npy')
    lena_PRECORNERDETECT_FLOAT32 = np.load('data/lena_PRECORNERDETECT_FLOAT32.npy')
    pcd8 = cvPreCornerDetect(lena_GRAY_U8)
    pcd32 = cvPreCornerDetect(lena_GRAY_FLOAT32)
    assert_array_almost_equal(pcd8, lena_PRECORNERDETECT_U8)
    assert_array_almost_equal(pcd32, lena_PRECORNERDETECT_FLOAT32)
    assert_raises(Exception, cvPreCornerDetect, lena_RGB_U8)
    assert_raises(Exception, cvPreCornerDetect, (lena_GRAY_U8, lena_GRAY_U8))
    assert_raises(Exception, cvPreCornerDetect, (lena_GRAY_U8, lena_PRECORNERDETECT_U8))
    assert_raises(Exception, cvPreCornerDetect, lena_GRAY_U8, {'aperture_size':1})
    assert_raises(Exception, cvPreCornerDetect, lena_GRAY_U8, {'aperture_size':2})
    assert_raises(Exception, cvPreCornerDetect, lena_GRAY_U8, {'aperture_size':4})
    assert_raises(Exception, cvPreCornerDetect, lena_GRAY_U8, {'aperture_size':6})
    assert_raises(Exception, cvPreCornerDetect, lena_GRAY_U8, {'aperture_size':8})
    
@dec.skipif(OPENCV_LIBS_NOTFOUND, 'Skipping OpenCV test because OpenCV libs were not found')   
def test_cvCornerEigenValsAndVecs():
    lena_RGB_U8 = np.load('data/lena_RGB_U8.npy')
    lena_GRAY_U8 = np.load('data/lena_GRAY_U8.npy')
    lena_GRAY_FLOAT32 = np.load('data/lena_GRAY_FLOAT32.npy')
    lena_CEVAV_U8 = np.load('data/lena_CEVAV_U8.npy')
    lena_CEVAV_FLOAT32 = np.load('data/lena_CEVAV_FLOAT32.npy')
    cevav8 = cvCornerEigenValsAndVecs(lena_GRAY_U8)
    cevav32 = cvCornerEigenValsAndVecs(lena_GRAY_FLOAT32)
    assert_array_almost_equal(cevav8, lena_CEVAV_U8)
    assert_array_almost_equal(cevav32, lena_CEVAV_FLOAT32)
    assert_raises(Exception, cvCornerEigenValsAndVecs, lena_RGB_U8)
    assert_raises(Exception, cvCornerEigenValsAndVecs, lena_GRAY_U8, {'aperture_size':1})
    assert_raises(Exception, cvCornerEigenValsAndVecs, lena_GRAY_U8, {'aperture_size':2})
    assert_raises(Exception, cvCornerEigenValsAndVecs, lena_GRAY_U8, {'aperture_size':4})
    assert_raises(Exception, cvCornerEigenValsAndVecs, lena_GRAY_U8, {'aperture_size':6})
    assert_raises(Exception, cvCornerEigenValsAndVecs, lena_GRAY_U8, {'aperture_size':8})
        
@dec.skipif(OPENCV_LIBS_NOTFOUND, 'Skipping OpenCV test because OpenCV libs were not found')   
def test_cvCornerMinEigenVal():
    lena_RGB_U8 = np.load('data/lena_RGB_U8.npy')
    lena_GRAY_U8 = np.load('data/lena_GRAY_U8.npy')
    lena_GRAY_FLOAT32 = np.load('data/lena_GRAY_FLOAT32.npy')
    lena_CMEV_U8 = np.load('data/lena_CMEV_U8.npy')
    lena_CMEV_FLOAT32 = np.load('data/lena_CMEV_FLOAT32.npy')
    cmev8 = cvCornerMinEigenVal(lena_GRAY_U8)
    cmev32 = cvCornerMinEigenVal(lena_GRAY_FLOAT32)
    assert_array_almost_equal(cmev8, lena_CMEV_U8)
    assert_array_almost_equal(cmev32, lena_CMEV_FLOAT32)
    assert_raises(Exception, cvCornerMinEigenVal, lena_RGB_U8)
    assert_raises(Exception, cvCornerMinEigenVal, lena_GRAY_U8, {'aperture_size':1})
    assert_raises(Exception, cvCornerMinEigenVal, lena_GRAY_U8, {'aperture_size':2})
    assert_raises(Exception, cvCornerMinEigenVal, lena_GRAY_U8, {'aperture_size':4})
    assert_raises(Exception, cvCornerMinEigenVal, lena_GRAY_U8, {'aperture_size':6})
    assert_raises(Exception, cvCornerMinEigenVal, lena_GRAY_U8, {'aperture_size':8})

@dec.skipif(OPENCV_LIBS_NOTFOUND, 'Skipping OpenCV test because OpenCV libs were not found')   
def test_cvCornerHarris():
    lena_RGB_U8 = np.load('data/lena_RGB_U8.npy')
    lena_GRAY_U8 = np.load('data/lena_GRAY_U8.npy')
    lena_GRAY_FLOAT32 = np.load('data/lena_GRAY_FLOAT32.npy')
    lena_HARRIS_U8 = np.load('data/lena_HARRIS_U8.npy')
    lena_HARRIS_FLOAT32 = np.load('data/lena_HARRIS_FLOAT32.npy')
    hc8 = cvCornerHarris(lena_GRAY_U8)
    hc32 = cvCornerHarris(lena_GRAY_FLOAT32)
    assert_array_almost_equal(hc8, lena_HARRIS_U8)
    assert_array_almost_equal(hc32, lena_HARRIS_FLOAT32)
    assert_raises(Exception, cvCornerHarris, lena_RGB_U8)
    assert_raises(Exception, cvCornerHarris, lena_GRAY_U8, {'aperture_size':1})
    assert_raises(Exception, cvCornerHarris, lena_GRAY_U8, {'aperture_size':2})
    assert_raises(Exception, cvCornerHarris, lena_GRAY_U8, {'aperture_size':4})
    assert_raises(Exception, cvCornerHarris, lena_GRAY_U8, {'aperture_size':6})
    assert_raises(Exception, cvCornerHarris, lena_GRAY_U8, {'aperture_size':8})
    
@dec.skipif(OPENCV_LIBS_NOTFOUND, 'Skipping OpenCV test because OpenCV libs were not found')
def test_cvSmooth():
    lena_RGB_U8 = np.load('data/lena_RGB_U8.npy')
    lena_GRAY_U8 = np.load('data/lena_GRAY_U8.npy')
    lena_GAUSSRGB_U8 = np.load('data/lena_GAUSSRGB_U8.npy')
    lena_GAUSSGRAY_U8 = np.load('data/lena_GAUSSGRAY_U8.npy')
    gaussrgb = cvSmooth(lena_RGB_U8)
    gaussgray = cvSmooth(lena_GRAY_U8)
    assert_array_equal(gaussrgb, lena_GAUSSRGB_U8)
    assert_array_equal(gaussgray, lena_GAUSSGRAY_U8)
    assert_raises(Exception, cvSmooth, (lena_RGB_U8), {'smoothtype': CV_BLUR_NO_SCALE})
    
                                                       
    
if __name__ == '__main__':
    run_module_suite()
