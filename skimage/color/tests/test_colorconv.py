#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for color conversion functions.

Authors
-------
- the rgb2hsv test was written by Nicolas Pinto, 2009
- other tests written by Ralf Gommers, 2009

:license: modified BSD
"""

import os.path

import numpy as np
from numpy.testing import *

from skimage import img_as_float
from skimage.io import imread
from skimage.color import (
    rgb2hsv, hsv2rgb,
    rgb2xyz, xyz2rgb,
    rgb2rgbcie, rgbcie2rgb,
    convert_colorspace,
    rgb2grey, gray2rgb,
    xyz2lab, lab2xyz,
    lab2rgb, rgb2lab
    )

from skimage import data_dir

import colorsys


class TestColorconv(TestCase):

    img_rgb = imread(os.path.join(data_dir, 'color.png'))
    img_grayscale = imread(os.path.join(data_dir, 'camera.png'))

    colbars = np.array([[1, 1, 0, 0, 1, 1, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 0, 1, 0, 1, 0, 1, 0]]).astype(np.float)
    colbars_array = np.swapaxes(colbars.reshape(3, 4, 2), 0, 2)
    colbars_point75 = colbars * 0.75
    colbars_point75_array = np.swapaxes(colbars_point75.reshape(3, 4, 2), 0, 2)

    xyz_array = np.array([[[0.4124, 0.21260, 0.01930]],  # red
                    [[0, 0, 0]],  # black
                    [[.9505, 1., 1.089]],  # white
                    [[.1805, .0722, .9505]],  # blue
                    [[.07719, .15438, .02573]],  # green
                    ])
    lab_array = np.array([[[53.233, 80.109, 67.220]],  # red
                    [[0., 0., 0.]],  # black
                    [[100.0, 0.005, -0.010]],  # white
                    [[32.303, 79.197, -107.864]],  # blue
                    [[46.229, -51.7, 49.898]],  # green
                    ])

    # RGB to HSV
    def test_rgb2hsv_conversion(self):
        rgb = img_as_float(self.img_rgb)[::16, ::16]
        hsv = rgb2hsv(rgb).reshape(-1, 3)
        # ground truth from colorsys
        gt = np.array([colorsys.rgb_to_hsv(pt[0], pt[1], pt[2])
                       for pt in rgb.reshape(-1, 3)]
                      )
        assert_almost_equal(hsv, gt)

    def test_rgb2hsv_error_grayscale(self):
        self.assertRaises(ValueError, rgb2hsv, self.img_grayscale)

    def test_rgb2hsv_error_one_element(self):
        self.assertRaises(ValueError, rgb2hsv, self.img_rgb[0, 0])

    # HSV to RGB
    def test_hsv2rgb_conversion(self):
        rgb = self.img_rgb.astype("float32")[::16, ::16]
        # create HSV image with colorsys
        hsv = np.array([colorsys.rgb_to_hsv(pt[0], pt[1], pt[2])
                        for pt in rgb.reshape(-1, 3)]).reshape(rgb.shape)
        # convert back to RGB and compare with original.
        # relative precision for RGB -> HSV roundtrip is about 1e-6
        assert_almost_equal(rgb, hsv2rgb(hsv), decimal=4)

    def test_hsv2rgb_error_grayscale(self):
        self.assertRaises(ValueError, hsv2rgb, self.img_grayscale)

    def test_hsv2rgb_error_one_element(self):
        self.assertRaises(ValueError, hsv2rgb, self.img_rgb[0, 0])

    # RGB to XYZ
    def test_rgb2xyz_conversion(self):
        gt = np.array([[[0.950456, 1.      , 1.088754],
                        [0.538003, 0.787329, 1.06942 ],
                        [0.592876, 0.28484 , 0.969561],
                        [0.180423, 0.072169, 0.950227]],
                       [[0.770033, 0.927831, 0.138527],
                        [0.35758 , 0.71516 , 0.119193],
                        [0.412453, 0.212671, 0.019334],
                        [0.      , 0.      , 0.      ]]])
        assert_almost_equal(rgb2xyz(self.colbars_array), gt)

    # stop repeating the "raises" checks for all other functions that are
    # implemented with color._convert()
    def test_rgb2xyz_error_grayscale(self):
        self.assertRaises(ValueError, rgb2xyz, self.img_grayscale)

    def test_rgb2xyz_error_one_element(self):
        self.assertRaises(ValueError, rgb2xyz, self.img_rgb[0, 0])

    # XYZ to RGB
    def test_xyz2rgb_conversion(self):
        # only roundtrip test, we checked rgb2xyz above already
        assert_almost_equal(xyz2rgb(rgb2xyz(self.colbars_array)),
                            self.colbars_array)

    # RGB to RGB CIE
    def test_rgb2rgbcie_conversion(self):
        gt = np.array([[[ 0.1488856 ,  0.18288098,  0.19277574],
                        [ 0.01163224,  0.16649536,  0.18948516],
                        [ 0.12259182,  0.03308008,  0.17298223],
                        [-0.01466154,  0.01669446,  0.16969164]],
                       [[ 0.16354714,  0.16618652,  0.0230841 ],
                        [ 0.02629378,  0.1498009 ,  0.01979351],
                        [ 0.13725336,  0.01638562,  0.00329059],
                        [ 0.        ,  0.        ,  0.        ]]])
        assert_almost_equal(rgb2rgbcie(self.colbars_array), gt)

    # RGB CIE to RGB
    def test_rgbcie2rgb_conversion(self):
        # only roundtrip test, we checked rgb2rgbcie above already
        assert_almost_equal(rgbcie2rgb(rgb2rgbcie(self.colbars_array)),
                            self.colbars_array)

    def test_convert_colorspace(self):
        colspaces = ['HSV', 'RGB CIE', 'XYZ']
        colfuncs_from = [hsv2rgb, rgbcie2rgb, xyz2rgb]
        colfuncs_to = [rgb2hsv, rgb2rgbcie, rgb2xyz]

        assert_almost_equal(convert_colorspace(self.colbars_array, 'RGB',
                                               'RGB'), self.colbars_array)
        for i, space in enumerate(colspaces):
            gt = colfuncs_from[i](self.colbars_array)
            assert_almost_equal(convert_colorspace(self.colbars_array, space,
                                                  'RGB'), gt)
            gt = colfuncs_to[i](self.colbars_array)
            assert_almost_equal(convert_colorspace(self.colbars_array, 'RGB',
                                                   space), gt)

        self.assertRaises(ValueError, convert_colorspace, self.colbars_array,
                                                           'nokey', 'XYZ')
        self.assertRaises(ValueError, convert_colorspace, self.colbars_array,
                                                           'RGB', 'nokey')

    def test_rgb2grey(self):
        x = np.array([1, 1, 1]).reshape((1, 1, 3)).astype(np.float)
        g = rgb2grey(x)
        assert_array_almost_equal(g, 1)

        assert_equal(g.shape, (1, 1))

    def test_rgb2grey_on_grey(self):
        rgb2grey(np.random.random((5, 5)))

    # test matrices for xyz2lab and lab2xyz generated using http://www.easyrgb.com/index.php?X=CALC
    # Note: easyrgb website displays xyz*100
    def test_xyz2lab(self):
        assert_array_almost_equal(xyz2lab(self.xyz_array),
                                  self.lab_array, decimal=3)

    def test_lab2xyz(self):
        assert_array_almost_equal(lab2xyz(self.lab_array),
                                  self.xyz_array, decimal=3)

    def test_lab_rgb_roundtrip(self):
        img_rgb = img_as_float(self.img_rgb)
        assert_array_almost_equal(lab2rgb(rgb2lab(img_rgb)), img_rgb)


def test_gray2rgb():
    x = np.array([0, 0.5, 1])
    assert_raises(ValueError, gray2rgb, x)

    x = x.reshape((3, 1))
    y = gray2rgb(x)

    assert_equal(y.shape, (3, 1, 3))
    assert_equal(y.dtype, x.dtype)

    x = np.array([[0, 128, 255]], dtype=np.uint8)
    z = gray2rgb(x)

    assert_equal(z.shape, (1, 3, 3))
    assert_equal(z[..., 0], x)
    assert_equal(z[0, 1, :], [128, 128, 128])

if __name__ == "__main__":
    run_module_suite()
