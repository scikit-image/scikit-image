#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
:author: Nicolas Pinto, 2009
:license: modified BSD
"""

from os import path

import numpy as np
from numpy.testing import *

from scikits.image.io import imread
from scikits.image.color import (
    rgb2hsv,
    )

from scikits.image import data

import colorsys

class TestColorconv(TestCase):

    img_rgb = imread(path.join(data.MY_PATH, 'color.png'))
    img_grayscale = imread(path.join(data.MY_PATH, 'camera.png'))

    def test_rgb2hsv_conversion(self):
        rgb = self.img_rgb.astype("float32")[::16, ::16]
        hsv = rgb2hsv(rgb).reshape(-1, 3)
        # ground truth from colorsys
        gt = np.array([colorsys.rgb_to_hsv(pt[0], pt[1], pt[2])
                       for pt in rgb.reshape(-1, 3)]
                      )
        assert_almost_equal(hsv, gt)

    def test_rgb2hsv_error_grayscale(self):        
        self.assertRaises(ValueError, rgb2hsv, self.img_grayscale)

    def test_rgb2hsv_error_one_element(self):
        self.assertRaises(ValueError, rgb2hsv, self.img_rgb[0,0])

    def test_rgb2hsv_error_list(self):
        self.assertRaises(TypeError, rgb2hsv, [self.img_rgb[0,0]])

if __name__ == "__main__":
    run_module_suite()

