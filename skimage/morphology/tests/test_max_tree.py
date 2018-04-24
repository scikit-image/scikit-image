import math
import unittest

import numpy as np
from skimage.morphology import max_tree
from skimage.morphology import extrema
from skimage.measure import label
from scipy import ndimage as ndi

import skimage.io
import time
import pdb

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

#     def test_diameter_closing(self):
#         "Test for Diameter Closing (2 diameters, all types)"
# 
#         # original image
#         img = np.array(
#             [[240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
#              [240, 200, 200, 240, 200, 240, 200, 200, 240, 240, 200, 240],
#              [240, 200, 40, 240, 240, 240, 240, 240, 240, 240, 40, 240],
#              [240, 240, 240, 240, 100, 240, 100, 100, 240, 240, 200, 240],
#              [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
#              [200, 200, 200, 200, 200, 200, 200, 240, 200, 200, 255, 255],
#              [200, 255, 200, 200, 200, 255, 200, 240, 255, 255, 255, 40],
#              [200, 200, 200, 100, 200, 200, 200, 240, 255, 255, 255, 255],
#              [200, 200, 200, 100, 200, 200, 200, 240, 200, 200, 255, 255],
#              [200, 200, 200, 200, 200, 40, 200, 240, 240, 100, 255, 255],
#              [200, 40, 255, 255, 255, 40, 200, 255, 200, 200, 255, 255],
#              [200, 200, 200, 200, 200, 200, 200, 255, 255, 255, 255, 255]],
#             dtype=np.uint8)
# 
#         # expected diameter closing with diameter 2
#         expected_2 = np.array(
#             [[240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
#              [240, 200, 200, 240, 240, 240, 200, 200, 240, 240, 200, 240],
#              [240, 200, 200, 240, 240, 240, 240, 240, 240, 240, 200, 240],
#              [240, 240, 240, 240, 240, 240, 100, 100, 240, 240, 200, 240],
#              [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
#              [200, 200, 200, 200, 200, 200, 200, 240, 200, 200, 255, 255],
#              [200, 255, 200, 200, 200, 255, 200, 240, 255, 255, 255, 255],
#              [200, 200, 200, 100, 200, 200, 200, 240, 255, 255, 255, 255],
#              [200, 200, 200, 100, 200, 200, 200, 240, 200, 200, 255, 255],
#              [200, 200, 200, 200, 200, 40, 200, 240, 240, 200, 255, 255],
#              [200, 200, 255, 255, 255, 40, 200, 255, 200, 200, 255, 255],
#              [200, 200, 200, 200, 200, 200, 200, 255, 255, 255, 255, 255]],
#             dtype=np.uint8)
# 
#         # expected diameter closing with diameter 3
#         expected_3 = np.array(
#             [[240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
#              [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 200, 240],
#              [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 200, 240],
#              [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 200, 240],
#              [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
#              [200, 200, 200, 200, 200, 200, 200, 240, 240, 240, 255, 255],
#              [200, 255, 200, 200, 200, 255, 200, 240, 255, 255, 255, 255],
#              [200, 200, 200, 200, 200, 200, 200, 240, 255, 255, 255, 255],
#              [200, 200, 200, 200, 200, 200, 200, 240, 200, 200, 255, 255],
#              [200, 200, 200, 200, 200, 200, 200, 240, 240, 200, 255, 255],
#              [200, 200, 255, 255, 255, 200, 200, 255, 200, 200, 255, 255],
#              [200, 200, 200, 200, 200, 200, 200, 255, 255, 255, 255, 255]],
#             dtype=np.uint8)
# 
#         # _full_type_test makes a test with many image types.
#         _full_type_test(img, 2, expected_2, max_tree.diameter_closing)
#         _full_type_test(img, 3, expected_3, max_tree.diameter_closing)

    def test_max_tree(self):
        "Test for max tree"
        img_type = np.uint8
        img = np.array([[10, 8,  8,  9],
                        [7,  7,  9,  9],
                        [8,  7, 10, 10],
                        [9,  9, 10, 10]], dtype=img_type)

        P_exp = np.array([[ 1,  4,  1,  1],
                          [ 4,  4,  3,  3],
                          [ 1,  4,  3, 10],
                          [ 3,  3, 10, 10]], dtype=np.int64)

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
        img = np.array([[ 15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15],
                        [ 15,  55,  55,  15,  55,  15,  55,  55,  15,  15,  55,  15],
                        [ 15,  55, 215,  15,  15,  15,  15,  15,  15,  15, 215,  15],
                        [ 15,  15,  15,  15, 155,  15, 155, 155,  15,  15,  55,  15],
                        [ 15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15],
                        [ 55,  55,  55,  55,  55,  55,  55,  15,  55,  55,   0,   0],
                        [ 55,   0,  55,  55,  55,   0,  55,  15,   0,   0,   0, 215],
                        [ 55,  55,  55, 155,  55,  55,  55,  15,   0,   0,   0,   0],
                        [ 55,  55,  55, 155,  55,  55,  55,  15,  55,  55,   0,   0],
                        [ 55,  55,  55,  55,  55, 215,  55,  15,  15, 155,   0,   0],
                        [ 55, 215,   0,   0,   0, 215,  55,   0,  55,  55,   0,   0],
                        [ 55,  55,  55,  55,  55,  55,  55,   0,   0,   0,   0,   0]], 
                        dtype=np.uint8)

        # expected area closing with area 2
        expected_2 = np.array([[ 15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15],
                               [ 15,  55,  55,  15,  15,  15,  55,  55,  15,  15,  55,  15],
                               [ 15,  55,  55,  15,  15,  15,  15,  15,  15,  15,  55,  15],
                               [ 15,  15,  15,  15,  15,  15, 155, 155,  15,  15,  55,  15],
                               [ 15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15],
                               [ 55,  55,  55,  55,  55,  55,  55,  15,  55,  55,   0,   0],
                               [ 55,   0,  55,  55,  55,   0,  55,  15,   0,   0,   0,   0],
                               [ 55,  55,  55, 155,  55,  55,  55,  15,   0,   0,   0,   0],
                               [ 55,  55,  55, 155,  55,  55,  55,  15,  55,  55,   0,   0],
                               [ 55,  55,  55,  55,  55, 215,  55,  15,  15,  55,   0,   0],
                               [ 55,  55,   0,   0,   0, 215,  55,   0,  55,  55,   0,   0],
                               [ 55,  55,  55,  55,  55,  55,  55,   0,   0,   0,   0,   0]], 
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


if __name__ == "__main__":
    np.testing.run_module_suite()
