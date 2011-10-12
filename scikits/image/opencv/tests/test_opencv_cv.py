# test for the opencv_cv extension module

from __future__ import with_statement

import os
import sys
import warnings

import numpy as np
from numpy.testing import *

from scikits.image import data_dir

if sys.version_info[0] < 3:
    import cPickle
else:
    import pickle as cPickle

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from scikits.image.opencv import *

opencv_skip = dec.skipif(not loaded, 'OpenCV libraries not found')

class OpenCVTest(object):
    lena_RGB_U8 = np.load(os.path.join(data_dir, 'lena_RGB_U8.npz'))['arr_0']
    lena_GRAY_U8 = np.load(os.path.join(data_dir, 'lena_GRAY_U8.npz'))['arr_0']


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
        chessboard_GRAY_U8 = np.load(
            os.path.join(data_dir, 'chessboard_GRAY_U8.npz')['arr_0'])
        pts = cvFindChessboardCorners(chessboard_GRAY_U8, (7, 7))


class TestDrawChessboardCorners(object):
    @opencv_skip
    def test_cvDrawChessboardCorners(self):
        chessboard_GRAY_U8 = np.load(
            os.path.join(data_dir, 'chessboard_GRAY_U8.npz')['arr_0'])
        chessboard_RGB_U8 = np.load(
            os.path.join(data_dir, 'chessboard_RGB_U8.npz')['arr_0'])
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
        _, (object_points,points_count,image_points,intrinsics,distortions) = \
           np.load(os.path.join(data_dir, "cvCalibrateCamera2TestData.npz"))

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


@opencv_skip
def test_cvFindFundamentalMat():
    #
    #  c2--->*        * = Data Cloud
    #        ^
    #        |           ^ z-direction
    #        c1       <--|
    #                 x
    #
    # Experimental setup: camera 1 at the origin, random cube data set in front,
    # camera two watching from the side (position [10, 0, 10])

    # Set up projection matrices

    def build_proj_mat(K, R, C):
        """
        Construct a projection matrix.

        Parameters
        ----------
        K : ndarray, 3x3
            Camera matrix, intrinsic parameters.
        R : ndarray, 3x3
            Rotation, world to camera.
        C : ndarray, (3,)
            Location of camera center in world coordinates.

        """
        C = np.reshape(C, (3, 1))

        KR = np.dot(K, R)
        P = np.zeros((3, 4))
        P[:3, :3] = KR
        P[:, 3].flat = np.dot(KR, -C)

        return P

    def cross_matrix(v):
        a = v[0]
        b = v[1]
        c = v[2]

        return np.array([[ 0, -c,  b],
                         [ c,  0, -a],
                         [-b,  a,  0]])

    # Camera one, at origin of world coordinates, looking down the z-axis
    K = np.array([[100., 0,   100],
                  [0,    100, 100],
                  [0,    0,   1]])
    R = np.eye(3)
    C = np.zeros((3,))
    P = build_proj_mat(K, R, C)

    # Camera two
    K_ = K
    R_ = np.array([[0., 0, -1],
                   [0,  1,  0],
                   [1,  0,  0]]) # Rotation of 90 degrees around y-axis
    C_ = np.array([[10., 0, 10]]).T
    P_ = build_proj_mat(K_, R_, C_)

    data = np.random.random((100, 4)) * 5 - 2.5
    data[:, 2] += 10 # Offset data in the z direction
    data[:, 3] = 1 # 4D homogeneous version of 3D coords

    points1 = np.dot(data, P.T)
    points2 = np.dot(data, P_.T)

    # See Hartley & Zisserman, Multiple View Geometry (2nd ed), p. 244
    t = -np.dot(R_, C_)
    K_t = np.dot(K_, t)

    # Under numpy >= 1.5, this would be:
    #F = cross_matrix(K_t).dot(K_).dot(R).dot(np.linalg.inv(K))

    F = np.dot(np.dot(np.dot(cross_matrix(K_t), K_), R_), np.linalg.inv(K))
    F /= F[2, 2]

    F_est, status = cvFindFundamentalMat(points1, points2)

    # Compare
    assert_array_almost_equal(F, F_est)

if __name__ == '__main__':
    run_module_suite()
