# test for the opencv_cv extension module

from __future__ import with_statement

import os
import warnings

import numpy as np
from numpy.testing import *

from scikits.image import data_dir
import cPickle

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from scikits.image.opencv import *

opencv_skip = dec.skipif(not loaded, 'OpenCV libraries not found')

class OpenCVTest(object):
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


class TestFindCornerSubPix(object):
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
        cvFindCornerSubPix(img, corners, (2, 2))


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
        cvResize(self.lena_RGB_U8, (50, 50), method=CV_INTER_LINEAR)
        cvResize(self.lena_RGB_U8, (200, 200), method=CV_INTER_CUBIC)


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


class TestLogPolar(OpenCVTest):
    @opencv_skip
    def test_cvLogPolar(self):
        img = self.lena_RGB_U8
        width = img.shape[1]
        height = img.shape[0]
        x = width / 2.
        y = height / 2.
        cvLogPolar(img, (x, y), 20)


class TestErode(OpenCVTest):
    @opencv_skip
    def test_cvErode(self):
        kern = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype='int32')
        cvErode(self.lena_RGB_U8, kern, in_place=True)


class TestDilate(OpenCVTest):
    @opencv_skip
    def test_cvDilate(self):
        kern = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype='int32')
        cvDilate(self.lena_RGB_U8, kern, in_place=True)


class TestMorphologyEx(OpenCVTest):
    @opencv_skip
    def test_cvMorphologyEx(self):
        kern = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype='int32')
        cvMorphologyEx(self.lena_RGB_U8, kern, CV_MOP_TOPHAT, in_place=True)


class TestSmooth(OpenCVTest):
    @opencv_skip
    def test_cvSmooth(self):
        for st in (CV_BLUR_NO_SCALE, CV_BLUR, CV_GAUSSIAN, CV_MEDIAN,
                   CV_BILATERAL):
            cvSmooth(self.lena_GRAY_U8, st, 3, 0, 0, 0, False)


class TestFilter2D(OpenCVTest):
    @opencv_skip
    def test_cvFilter2D(self):
        kern = np.array([[0, 1.5, 0],
                         [1, 1, 2.6],
                         [0, .76, 0]], dtype='float32')
        cvFilter2D(self.lena_RGB_U8, kern, in_place=True)


class TestIntegral(OpenCVTest):
    @opencv_skip
    def test_cvIntegral(self):
        cvIntegral(self.lena_RGB_U8, True, True)


class TestCvtColor(OpenCVTest):
    @opencv_skip
    def test_cvCvtColor(self):
        cvCvtColor(self.lena_RGB_U8, CV_RGB2BGR)
        cvCvtColor(self.lena_RGB_U8, CV_RGB2BGRA)
        cvCvtColor(self.lena_RGB_U8, CV_RGB2HSV)
        cvCvtColor(self.lena_RGB_U8, CV_RGB2BGR565)
        cvCvtColor(self.lena_RGB_U8, CV_RGB2BGR555)
        cvCvtColor(self.lena_RGB_U8, CV_RGB2GRAY)
        cvCvtColor(self.lena_GRAY_U8, CV_GRAY2BGR)
        cvCvtColor(self.lena_GRAY_U8, CV_GRAY2BGR565)
        cvCvtColor(self.lena_GRAY_U8, CV_GRAY2BGR555)


class TestThreshold(OpenCVTest):
    @opencv_skip
    def test_cvThreshold(self):
        cvThreshold(self.lena_GRAY_U8, 100, 255, CV_THRESH_BINARY)
        cvThreshold(self.lena_GRAY_U8, 100, 255, CV_THRESH_BINARY_INV)
        cvThreshold(self.lena_GRAY_U8, 100, threshold_type=CV_THRESH_TRUNC)
        cvThreshold(self.lena_GRAY_U8, 100, threshold_type=CV_THRESH_TOZERO)
        cvThreshold(self.lena_GRAY_U8, 100, threshold_type=CV_THRESH_TOZERO_INV)
        cvThreshold(self.lena_GRAY_U8, 100, 1, CV_THRESH_BINARY, use_otsu=True)


class TestAdaptiveThreshold(OpenCVTest):
    @opencv_skip
    def test_cvAdaptiveThreshold(self):
        cvAdaptiveThreshold(self.lena_GRAY_U8, 100)


class TestPyrDown(OpenCVTest):
    @opencv_skip
    def test_cvPyrDown(self):
        cvPyrDown(self.lena_RGB_U8)


class TestPyrUp(OpenCVTest):
    @opencv_skip
    def test_cvPyrUp(self):
        cvPyrUp(self.lena_RGB_U8)


class TestFindChessboardCorners(object):
    @opencv_skip
    def test_cvFindChessboardCorners(self):
        chessboard_GRAY_U8 = np.load(os.path.join(data_dir,
                                                  'chessboard_GRAY_U8.npy'))
        pts = cvFindChessboardCorners(chessboard_GRAY_U8, (7, 7))


class TestDrawChessboardCorners(object):
    @opencv_skip
    def test_cvDrawChessboardCorners(self):
        chessboard_GRAY_U8 = np.load(os.path.join(data_dir,
                                                  'chessboard_GRAY_U8.npy'))
        chessboard_RGB_U8 = np.load(os.path.join(data_dir,
                                                  'chessboard_RGB_U8.npy'))
        corners = cvFindChessboardCorners(chessboard_GRAY_U8, (7, 7))
        cvDrawChessboardCorners(chessboard_RGB_U8, (7, 7), corners)


class TestCalibrateCamera2(object):
    @opencv_skip
    def test_cvCalibrateCamera2_Identity(self):
        ys = xs = range(4)

        image_points = np.array( [(4 * x, 4 * y) for x in xs for y in ys ],
                dtype=np.float64)
        object_points = np.array( [(x, y, 0) for x in xs for y in ys ],
                dtype=np.float64)

        image_points = np.ascontiguousarray(np.vstack((image_points,) * 3))
        object_points = np.ascontiguousarray(np.vstack((object_points,) * 3))

        intrinsics, distortions = cvCalibrateCamera2(
            object_points, image_points,
            np.array([16, 16, 16], dtype=np.int32), (4, 4)
        )

        assert_almost_equal(distortions, np.array([0., 0., 0., 0., 0.]))
        # The intrinsics will be strange, but we can at least check
        # for known zeros and ones
        assert_almost_equal( intrinsics[0,1], 0)
        assert_almost_equal( intrinsics[1,0], 0)
        assert_almost_equal( intrinsics[2,0], 0)
        assert_almost_equal( intrinsics[2,1], 0)
        assert_almost_equal( intrinsics[2,2], 1)

    @opencv_skip
    @dec.slow
    def test_cvCalibrateCamera2_KnownData(self):
        (object_points,points_count,image_points,intrinsics,distortions) =\
             cPickle.load(open(os.path.join(
                 data_dir, "cvCalibrateCamera2TestData.pck"), "rb")
             )

        intrinsics_test, distortion_test = cvCalibrateCamera2(
            object_points, image_points, points_count, (1024,1280)
        )


class TestUndistort2(OpenCVTest):
    @opencv_skip
    def test_cvUndistort2(self):
        intrinsics = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]], dtype='float64')
        distortions = np.array([0., 0., 0., 0., 0.], dtype='float64')

        undist = cvUndistort2(self.lena_RGB_U8, intrinsics, distortions)
        undistg = cvUndistort2(self.lena_GRAY_U8, intrinsics, distortions)

        assert_array_almost_equal(undist, self.lena_RGB_U8)
        assert_array_almost_equal(undistg, self.lena_GRAY_U8)

    @opencv_skip
    def test_cvUndistort2_new_intrinsics(self):
        intrinsics = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]], dtype='float64')
        distortions = np.array([0., 0., 0., 0., 0.], dtype='float64')

        undist = cvUndistort2(self.lena_RGB_U8, intrinsics, distortions,
                              intrinsics)
        undistg = cvUndistort2(self.lena_GRAY_U8, intrinsics, distortions,
                               intrinsics)

        assert_array_almost_equal(undist, self.lena_RGB_U8)
        assert_array_almost_equal(undistg, self.lena_GRAY_U8)




if __name__ == '__main__':
    run_module_suite()
