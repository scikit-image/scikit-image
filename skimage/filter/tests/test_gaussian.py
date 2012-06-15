import unittest
import numpy as np
import skimage.filter as filter
import skimage.data as data
import skimage.io as io
import skimage
import os

def norm_float(image):
    imin = np.min(image)
    if imin > 0.:
      image -= imin
    return image / np.max(image)

def load(filename):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
    return io.imread(path)

class TestGaussian(unittest.TestCase):
    def test_gaussian_circular(self):
        '''Test blurring the image with circular gaussian kernel'''
        image = data.camera()
        blurred = filter.gaussian(image, 2.5)
        blurred_ubyte = skimage.img_as_ubyte(norm_float(blurred)) 
        control = load('camara_gaussian_blur_2_5.png')
        self.assertTrue(np.all(np.abs(blurred_ubyte - control) < 0.001))  

    def test_gaussian_ellips(self):
        '''Test blurring the image with ellipsoidal gaussian kernel'''
        image = data.camera()
        blurred = filter.gaussian(image, (2.5, 0.5))
        blurred_ubyte = skimage.img_as_ubyte(norm_float(blurred))
        control = load('camara_gaussian_blur_2_5_0_5.png')
        self.assertTrue(np.all(np.abs(blurred_ubyte - control) < 0.001))

