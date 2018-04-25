import math
import unittest

import numpy as np
from skimage.morphology import max_tree

eps = 1e-12


def diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    t = ((a - b)**2).sum()
    return math.sqrt(t)


def _full_type_test(img, param, expected, func, param_scale=False):

    # images as they are
    out = func(img, param)
    error = diff(out, expected)
    assert error < eps

    # unsigned int
    for dt in [np.uint32, np.uint64]:
        img_cast = img.astype(dt)
        out = func(img_cast, param)
        exp_cast = expected.astype(dt)
        error = diff(out, exp_cast)
        assert error < eps

    # float
    data_float = img.astype(np.float64)
    data_float = data_float / 255.0
    expected_float = expected.astype(np.float64)
    expected_float = expected_float / 255.0
    if param_scale:
        param_cast = param / 255.0
    else:
        param_cast = param
    for dt in [np.float32, np.float64]:
        data_cast = data_float.astype(dt)
        out = func(data_cast, param_cast)
        exp_cast = expected_float.astype(dt)
        error_img = 255.0 * exp_cast - 255.0 * out
        error = (error_img >= 1.0).sum()
        assert error < eps

    # signed images
    img_signed = img.astype(np.int16)
    img_signed = img_signed - 128
    exp_signed = expected.astype(np.int16)
    exp_signed = exp_signed - 128
    for dt in [np.int8, np.int16, np.int32, np.int64]:
        img_s = img_signed.astype(dt)
        out = func(img_s, param)
        exp_s = exp_signed.astype(dt)
        error = diff(out, exp_s)
        assert error < eps


class TestMaxtree(unittest.TestCase):

    def test_max_tree(self):
        "Test for max tree"
        img_type = np.uint8
        img = np.array([[10, 8,  8,  9],
                        [7,  7,  9,  9],
                        [8,  7, 10, 10],
                        [9,  9, 10, 10]], dtype=img_type)

        P_exp = np.array([[1,  4,  1,  1],
                          [4,  4,  3,  3],
                          [1,  4,  3, 10],
                          [3,  3, 10, 10]], dtype=np.int64)

        S_exp = np.array([ 4,  5,  9,  1,  2,  8,  3,  6,  7, 12, 13,  0, 10, 11, 14, 15],
                         dtype=np.int64)

        for img_type in [np.uint8, np.uint16, np.uint32, np.uint64]:
            img = img.astype(img_type)
            P, S = max_tree.build_max_tree(img)
            error = diff(P, P_exp)
            assert error < eps
            error = diff(S, S_exp)
            assert error < eps

        for img_type in [np.int8, np.int16, np.int32, np.int64]:
            img = img.astype(img_type)
            img_shifted = img - 9
            P, S = max_tree.build_max_tree(img_shifted)
            error = diff(P, P_exp)
            assert error < eps
            error = diff(S, S_exp)
            assert error < eps

        img_float = img.astype(np.float)
        img_float = (img_float-8) / 2.0
        for img_type in [np.float32, np.float64]:
            img_float = img_float.astype(img_type)
            P, S = max_tree.build_max_tree(img_float)
            error = diff(P, P_exp)
            assert error < eps
            error = diff(S, S_exp)
            assert error < eps

        return

    def test_area_closing(self):
        "Test for Area Closing (2 thresholds, all types)"

        # original image
        img = np.array(
            [[240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [240, 200, 200, 240, 200, 240, 200, 200, 240, 240, 200, 240],
             [240, 200, 40, 240, 240, 240, 240, 240, 240, 240, 40, 240],
             [240, 240, 240, 240, 100, 240, 100, 100, 240, 240, 200, 240],
             [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [200, 200, 200, 200, 200, 200, 200, 240, 200, 200, 255, 255],
             [200, 255, 200, 200, 200, 255, 200, 240, 255, 255, 255, 40],
             [200, 200, 200, 100, 200, 200, 200, 240, 255, 255, 255, 255],
             [200, 200, 200, 100, 200, 200, 200, 240, 200, 200, 255, 255],
             [200, 200, 200, 200, 200, 40, 200, 240, 240, 100, 255, 255],
             [200, 40, 255, 255, 255, 40, 200, 255, 200, 200, 255, 255],
             [200, 200, 200, 200, 200, 200, 200, 255, 255, 255, 255, 255]],
            dtype=np.uint8)

        # expected area closing with area 2
        expected_2 = np.array(
            [[240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [240, 200, 200, 240, 240, 240, 200, 200, 240, 240, 200, 240],
             [240, 200, 200, 240, 240, 240, 240, 240, 240, 240, 200, 240],
             [240, 240, 240, 240, 240, 240, 100, 100, 240, 240, 200, 240],
             [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [200, 200, 200, 200, 200, 200, 200, 240, 200, 200, 255, 255],
             [200, 255, 200, 200, 200, 255, 200, 240, 255, 255, 255, 255],
             [200, 200, 200, 100, 200, 200, 200, 240, 255, 255, 255, 255],
             [200, 200, 200, 100, 200, 200, 200, 240, 200, 200, 255, 255],
             [200, 200, 200, 200, 200, 40, 200, 240, 240, 200, 255, 255],
             [200, 200, 255, 255, 255, 40, 200, 255, 200, 200, 255, 255],
             [200, 200, 200, 200, 200, 200, 200, 255, 255, 255, 255, 255]],
            dtype=np.uint8)

        # expected diameter closing with diameter 4
        expected_4 = np.array(
            [[240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [240, 200, 200, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [240, 200, 200, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [200, 200, 200, 200, 200, 200, 200, 240, 240, 240, 255, 255],
             [200, 255, 200, 200, 200, 255, 200, 240, 255, 255, 255, 255],
             [200, 200, 200, 200, 200, 200, 200, 240, 255, 255, 255, 255],
             [200, 200, 200, 200, 200, 200, 200, 240, 200, 200, 255, 255],
             [200, 200, 200, 200, 200, 200, 200, 240, 240, 200, 255, 255],
             [200, 200, 255, 255, 255, 200, 200, 255, 200, 200, 255, 255],
             [200, 200, 200, 200, 200, 200, 200, 255, 255, 255, 255, 255]],
            dtype=np.uint8)

        # _full_type_test makes a test with many image types.
        _full_type_test(img, 2, expected_2, max_tree.area_closing)
        _full_type_test(img, 4, expected_4, max_tree.area_closing)

    def test_area_opening(self):
        "Test for Area Opening (2 thresholds, all types)"

        # original image
        img = np.array([[15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
                        [15, 55, 55, 15, 55, 15, 55, 55, 15, 15, 55, 15],
                        [15, 55,215, 15, 15, 15, 15, 15, 15, 15,215, 15],
                        [15, 15, 15, 15,155, 15,155,155, 15, 15, 55, 15],
                        [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
                        [55, 55, 55, 55, 55, 55, 55, 15, 55, 55,  0,  0],
                        [55,  0, 55, 55, 55,  0, 55, 15,  0,  0,  0,215],
                        [55, 55, 55,155, 55, 55, 55, 15,  0,  0,  0,  0],
                        [55, 55, 55,155, 55, 55, 55, 15, 55, 55,  0,  0],
                        [55, 55, 55, 55, 55,215, 55, 15, 15,155,  0,  0],
                        [55,215,  0,  0,  0,215, 55,  0, 55, 55,  0,  0],
                        [55, 55, 55, 55, 55, 55, 55,  0,  0,  0,  0,  0]],
                       dtype=np.uint8)

        # expected area closing with area 2
        expected_2 = np.array([[15, 15, 15,  15,  15,  15,  15,  15,  15,  15,  15,  15],
                               [15, 55, 55,  15,  15,  15,  55,  55,  15,  15,  55,  15],
                               [15, 55, 55,  15,  15,  15,  15,  15,  15,  15,  55,  15],
                               [15, 15, 15,  15,  15,  15, 155, 155,  15,  15,  55,  15],
                               [15, 15, 15,  15,  15,  15,  15,  15,  15,  15,  15,  15],
                               [55, 55, 55,  55,  55,  55,  55,  15,  55,  55,   0,   0],
                               [55,  0, 55,  55,  55,   0,  55,  15,   0,   0,   0,   0],
                               [55, 55, 55, 155,  55,  55,  55,  15,   0,   0,   0,   0],
                               [55, 55, 55, 155,  55,  55,  55,  15,  55,  55,   0,   0],
                               [55, 55, 55,  55,  55, 215,  55,  15,  15,  55,   0,   0],
                               [55, 55,  0,   0,   0, 215,  55,   0,  55,  55,   0,   0],
                               [55, 55, 55,  55,  55,  55,  55,   0,   0,   0,   0,   0]],
                              dtype=np.uint8)

        # expected diameter closing with diameter 4
        expected_4 = np.array([[15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
                               [15, 55, 55, 15, 15, 15, 15, 15, 15, 15, 15, 15],
                               [15, 55, 55, 15, 15, 15, 15, 15, 15, 15, 15, 15],
                               [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
                               [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
                               [55, 55, 55, 55, 55, 55, 55, 15, 15, 15,  0,  0],
                               [55,  0, 55, 55, 55,  0, 55, 15,  0,  0,  0,  0],
                               [55, 55, 55, 55, 55, 55, 55, 15,  0,  0,  0,  0],
                               [55, 55, 55, 55, 55, 55, 55, 15, 55, 55,  0,  0],
                               [55, 55, 55, 55, 55, 55, 55, 15, 15, 55,  0,  0],
                               [55, 55,  0,  0,  0, 55, 55,  0, 55, 55,  0,  0],
                               [55, 55, 55, 55, 55, 55, 55,  0,  0,  0,  0,  0]],
                              dtype=np.uint8)

        # _full_type_test makes a test with many image types.
        _full_type_test(img, 2, expected_2, max_tree.area_opening)
        _full_type_test(img, 4, expected_4, max_tree.area_opening)

    def test_local_maxima(self):
        "local maxima for various data types"
        data = np.array([[10,  11,  13,  14,  14,  15,  14,  14,  13,  11],
                         [11,  13,  15,  16,  16,  16,  16,  16,  15,  13],
                         [13,  15,  40,  40,  18,  18,  18,  60,  60,  15],
                         [14,  16,  40,  40,  19,  19,  19,  60,  60,  16],
                         [14,  16,  18,  19,  19,  19,  19,  19,  18,  16],
                         [15,  16,  18,  19,  19,  20,  19,  19,  18,  16],
                         [14,  16,  18,  19,  19,  19,  19,  19,  18,  16],
                         [14,  16,  80,  80,  19,  19,  19, 100, 100,  16],
                         [13,  15,  80,  80,  18,  18,  18, 100, 100,  15],
                         [11,  13,  15,  16,  16,  16,  16,  16,  15,  13]],
                        dtype=np.uint8)
        expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                   dtype=np.uint8)
        for dtype in [np.uint8, np.uint64, np.int8, np.int64]:

            test_data = data.astype(dtype)
            out = max_tree.local_maxima(test_data)

            error = diff(expected_result, out)
            assert error < eps
            assert out.dtype == expected_result.dtype

    def test_extrema_float(self):
        "specific tests for float type"
        data = np.array([[0.10, 0.11, 0.13, 0.14, 0.14, 0.15, 0.14,
                          0.14, 0.13, 0.11],
                         [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16,
                          0.16, 0.15, 0.13],
                         [0.13, 0.15, 0.40, 0.40, 0.18, 0.18, 0.18,
                          0.60, 0.60, 0.15],
                         [0.14, 0.16, 0.40, 0.40, 0.19, 0.19, 0.19,
                          0.60, 0.60, 0.16],
                         [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19,
                          0.19, 0.18, 0.16],
                         [0.15, 0.182, 0.18, 0.19, 0.204, 0.20, 0.19,
                          0.19, 0.18, 0.16],
                         [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19,
                          0.19, 0.18, 0.16],
                         [0.14, 0.16, 0.80, 0.80, 0.19, 0.19, 0.19,
                          1.0,  1.0, 0.16],
                         [0.13, 0.15, 0.80, 0.80, 0.18, 0.18, 0.18,
                          1.0, 1.0, 0.15],
                         [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16,
                          0.16, 0.15, 0.13]],
                        dtype=np.float32)

        expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                   dtype=np.uint8)

        # test for local maxima
        out = max_tree.local_maxima(data)
        error = diff(expected_result, out)
        assert error < eps

    def test_3d(self):
        """tests the detection of maxima in 3D."""
        img = np.zeros((8, 8, 8), dtype=np.uint8)
        local_maxima = np.zeros((8, 8, 8), dtype=np.uint8)

        # first maximum: only one pixel
        img[1, 1:3, 1:3] = 100
        img[2, 2, 2] = 200
        img[3, 1:3, 1:3] = 100
        local_maxima[2, 2, 2] = 1

        # second maximum: three pixels in z-direction
        img[5:8, 1, 1] = 200
        local_maxima[5:8, 1, 1] = 1

        # third: two maxima in 0 and 3.
        img[0, 5:8, 5:8] = 200
        img[1, 6, 6] = 100
        img[2, 5:7, 5:7] = 200
        img[0:3, 5:8, 5:8] += 50
        local_maxima[0, 5:8, 5:8] = 1
        local_maxima[2, 5:7, 5:7] = 1

        # four : one maximum in the corner of the square
        img[6:8, 6:8, 6:8] = 200
        img[7, 7, 7] = 255
        local_maxima[7, 7, 7] = 1

        out = max_tree.local_maxima(img, 1)
        error = diff(local_maxima, out)
        assert error < eps


if __name__ == "__main__":
    np.testing.run_module_suite()
