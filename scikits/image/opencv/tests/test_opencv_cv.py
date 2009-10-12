# test for the opencv_cv extension module

import numpy as np
from numpy.testing import *

try:
    from scikits.image.opencv import *
    OPENCV_LIBS_NOTFOUND = False
except:
    OPENCV_LIBS_NOTFOUND = True

_opencv_skip = dec.skipif(OPENCV_LIBS_NOTFOUND, 
                           'Skipping OpenCV test because OpenCV'
                           'libs were not found')    

class OpenCVTest:
    # setup only works as a module level function
    def __init__(self):
        self.lena_RGB_U8 = np.load('data/lena_RGB_U8.npy')
        self.lena_GRAY_U8 = np.load('data/lena_GRAY_U8.npy')

        
class TestSobel(OpenCVTest):
    @_opencv_skip
    def test_cvSobel(self):
        cvSobel(self.lena_GRAY_U8)
        
        
class TestLaplace(OpenCVTest):
    @_opencv_skip
    def test_cvLaplace(self):
        cvLaplace(self.lena_GRAY_U8)
    
        
class TestCanny(OpenCVTest):
    @_opencv_skip
    def test_cvCanny(self):
        cvCanny(self.lena_GRAY_U8)

        
class TestPreCornerDetect(OpenCVTest):
    @_opencv_skip
    def test_cvPreCornerDetect(self):
        cvPreCornerDetect(self.lena_GRAY_U8)


class TestCornerEigenValsAndVecs(OpenCVTest):
    @_opencv_skip
    def test_cvCornerEigenValsAndVecs(self):
        cvCornerEigenValsAndVecs(self.lena_GRAY_U8)
        

class TestCornerMinEigenVal(OpenCVTest):
    @_opencv_skip
    def test_cvCornerMinEigenVal(self):
        cvCornerMinEigenVal(self.lena_GRAY_U8)
        

class TestCornerHarris(OpenCVTest):
    @_opencv_skip
    def test_cvCornerHarris(self):
        cvCornerHarris(self.lena_GRAY_U8)
        
        
class TestSmooth(OpenCVTest):
    @_opencv_skip
    def test_cvSmooth(self):
        for st in (CV_BLUR_NO_SCALE, CV_BLUR, CV_GAUSSIAN, CV_MEDIAN, 
                   CV_BILATERAL):
            cvSmooth(self.lena_GRAY_U8, None, st, 3, 0, 0, 0, False)
                  
    
if __name__ == '__main__':
    run_module_suite()
