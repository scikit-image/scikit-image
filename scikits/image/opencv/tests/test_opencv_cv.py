# test for the opencv_cv extension module
import os
import warnings

import numpy as np
from numpy.testing import *

from scikits.image import data_dir

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from scikits.image.opencv import *

opencv_skip = dec.skipif(cv is None,
                         'OpenCV libraries not found')

class OpenCVTest:
    lena_RGB_U8 = np.load(os.path.join(data_dir, 'lena_RGB_U8.npy'))
    lena_GRAY_U8 = np.load(os.path.join(data_dir, 'lena_GRAY_U8.npy'))
    

class TestSobel(OpenCVTest):
    @opencv_skip
    def test_cvSobel(self):
        cvSobel(self.lena_GRAY_U8)


class TestLaplace(OpenCVTest):
    @opencv_skip
    def test_cvLaplace(self):
        cvLaplace(self.lena_GRAY_U8)


class TestCanny(OpenCVTest):
    @opencv_skip
    def test_cvCanny(self):
        cvCanny(self.lena_GRAY_U8)


class TestPreCornerDetect(OpenCVTest):
    @opencv_skip
    def test_cvPreCornerDetect(self):
        cvPreCornerDetect(self.lena_GRAY_U8)


class TestCornerEigenValsAndVecs(OpenCVTest):
    @opencv_skip
    def test_cvCornerEigenValsAndVecs(self):
        cvCornerEigenValsAndVecs(self.lena_GRAY_U8)


class TestCornerMinEigenVal(OpenCVTest):
    @opencv_skip
    def test_cvCornerMinEigenVal(self):
        cvCornerMinEigenVal(self.lena_GRAY_U8)


class TestCornerHarris(OpenCVTest):
    @opencv_skip
    def test_cvCornerHarris(self):
        cvCornerHarris(self.lena_GRAY_U8)


class TestSmooth(OpenCVTest):
    @opencv_skip
    def test_cvSmooth(self):
        for st in (CV_BLUR_NO_SCALE, CV_BLUR, CV_GAUSSIAN, CV_MEDIAN,
                   CV_BILATERAL):
            cvSmooth(self.lena_GRAY_U8, None, st, 3, 0, 0, 0, False)
            

class TestFindCornerSubPix:
    @opencv_skip
    def test_cvFindCornersSubPix(self):
        img = np.array([[1, 1, 1, 0, 0, 0, 1, 1, 1],
                        [1, 1, 1, 0, 0, 0, 1, 1, 1],
                        [1, 1, 1, 0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 1, 1],
                        [1, 1, 1, 0, 0, 0, 1, 1, 1],
                        [1, 1, 1, 0, 0, 0, 1, 1, 1]], dtype='uint8')
        corners = np.array([[2, 2],
                            [2, 5],
                            [5, 2],
                            [5, 5]], dtype='float32')
        cvFindCornerSubPix(img, corners, 4, (2, 2))


class TestGoodFeaturesToTrack(OpenCVTest):
    @opencv_skip
    def test_cvGoodFeaturesToTrack(self):
        cvGoodFeaturesToTrack(self.lena_GRAY_U8, 100, 0.1, 3)
        

class TestGetRectSubPix(OpenCVTest):
    @opencv_skip
    def test_cvGetRectSubPix(self):
        cvGetRectSubPix(self.lena_RGB_U8, (20, 20), (48.6, 48.6))
        
        
class TestGetQuadrangleSubPix(OpenCVTest):
    @opencv_skip
    def test_cvGetQuadrangleSubPix(self):
        warpmat = np.array([[0.5, 0.3, 0.4],
                            [-.4, .23, 0.4]], dtype='float32')
        cvGetQuadrangleSubPix(self.lena_RGB_U8, warpmat)

        
class TestResize(OpenCVTest):
    @opencv_skip
    def test_cvResize(self):
        cvResize(self.lena_RGB_U8, height=50, width=50, method=CV_INTER_LINEAR)
        cvResize(self.lena_RGB_U8, height=200, width=200, method=CV_INTER_CUBIC)
        
        
class TestWarpAffine(OpenCVTest):
    @opencv_skip
    def test_cvWarpAffine(self):
        warpmat = np.array([[0.5, 0.3, 0.4],
                            [-.4, .23, 0.4]], dtype='float32')
        cvWarpAffine(self.lena_RGB_U8, warpmat)
        
        
class TestWarpPerspective(OpenCVTest):
    @opencv_skip
    def test_cvWarpPerspective(self):
        warpmat = np.array([[0.5, 0.3, 0.4],
                            [-.4, .23, 0.4],
                            [0.0, 1.0, 1.0]], dtype='float32')
        cvWarpPerspective(self.lena_RGB_U8, warpmat)
        
        
class TestFindChessboardCorners:
    @opencv_skip
    def test_cvFindChessboardCorners(self):
        chessboard_GRAY_U8 = np.load(os.path.join(data_dir, 
                                                  'chessboard_GRAY_U8.npy'))        
        pts = cvFindChessboardCorners(chessboard_GRAY_U8, (7, 7))      
        
    
class TestDrawChessboardCorners:
    @opencv_skip
    def test_cvDrawChessboardCorners(self):
        chessboard_GRAY_U8 = np.load(os.path.join(data_dir, 
                                                  'chessboard_GRAY_U8.npy'))        
        chessboard_RGB_U8 = np.load(os.path.join(data_dir, 
                                                  'chessboard_RGB_U8.npy'))                        
        corners = cvFindChessboardCorners(chessboard_GRAY_U8, (7, 7))        
        cvDrawChessboardCorners(chessboard_RGB_U8, (7, 7), corners)
        
        
        
if __name__ == '__main__':
    run_module_suite()
