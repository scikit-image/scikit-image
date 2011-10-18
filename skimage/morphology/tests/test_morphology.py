import os.path

import numpy as np
from numpy.testing import *

from skimage import data_dir
from skimage.io import imread
from skimage import data_dir
from skimage.morphology import *

lena = np.load(os.path.join(data_dir, 'lena_GRAY_U8.npy'))

class TestMorphology():

    def morph_worker(self, img, fn, morph_func, strel_func):
        matlab_results = np.load(os.path.join(data_dir, fn))
        k = 0
        for arrname in sorted(matlab_results):
            expected_result = matlab_results[arrname]
            mask = strel_func(k)
            actual_result = morph_func(lena, mask)
            assert_equal(expected_result, actual_result)
            k = k + 1

    def test_erode_diamond(self):
        self.morph_worker(lena, "diamond-erode-matlab-output.npz",
                          greyscale_erode, diamond)

    def test_dilate_diamond(self):
        self.morph_worker(lena, "diamond-dilate-matlab-output.npz",
                          greyscale_dilate, diamond)

    def test_open_diamond(self):
        self.morph_worker(lena, "diamond-open-matlab-output.npz",
                          greyscale_open, diamond)

    def test_close_diamond(self):
        self.morph_worker(lena, "diamond-close-matlab-output.npz",
                          greyscale_close, diamond)

    def test_tophat_diamond(self):
        self.morph_worker(lena, "diamond-tophat-matlab-output.npz",
                          greyscale_white_top_hat, diamond)

    def test_bothat_diamond(self):
        self.morph_worker(lena, "diamond-bothat-matlab-output.npz",
                          greyscale_black_top_hat, diamond)

    def test_erode_disk(self):
        self.morph_worker(lena, "disk-erode-matlab-output.npz",
                          greyscale_erode, disk)

    def test_dilate_disk(self):
        self.morph_worker(lena, "disk-dilate-matlab-output.npz",
                          greyscale_dilate, disk)

    def test_open_disk(self):
        self.morph_worker(lena, "disk-open-matlab-output.npz",
                          greyscale_open, disk)

    def test_close_disk(self):
        self.morph_worker(lena, "disk-close-matlab-output.npz",
                          greyscale_close, disk)


class TestEccentricStructuringElements():

    def setUp(self):
        self.black_pixel = 255 * np.ones((4, 4), dtype=np.uint8)
        self.black_pixel[1, 1] = 0
        self.white_pixel = 255 - self.black_pixel
        self.selems = [square(2), rectangle(2, 2),
                       rectangle(2, 1), rectangle(1, 2)]

    def test_dilate_erode_symmetry(self):
        for s in self.selems:
            c = greyscale_erode(self.black_pixel, s)
            d = greyscale_dilate(self.white_pixel, s)
            assert np.all(c == (255 - d))

    def test_open_black_pixel(self):
        for s in self.selems:
            grey_open = greyscale_open(self.black_pixel, s)
            assert np.all(grey_open == self.black_pixel)

    def test_close_white_pixel(self):
        for s in self.selems:
            grey_close = greyscale_close(self.white_pixel, s)
            assert np.all(grey_close == self.white_pixel)

    def test_open_white_pixel(self):
        for s in self.selems:
            assert np.all(greyscale_open(self.white_pixel, s) == 0)

    def test_close_black_pixel(self):
        for s in self.selems:
            assert np.all(greyscale_close(self.black_pixel, s) == 255)

    def test_white_tophat_white_pixel(self):
        for s in self.selems:
            tophat = greyscale_white_top_hat(self.white_pixel, s)
            assert np.all(tophat == self.white_pixel)

    def test_black_tophat_black_pixel(self):
        for s in self.selems:
            tophat = greyscale_black_top_hat(self.black_pixel, s)
            assert np.all(tophat == (255 - self.black_pixel))

    def test_white_tophat_black_pixel(self):
        for s in self.selems:
            tophat = greyscale_white_top_hat(self.black_pixel, s)
            assert np.all(tophat == 0)

    def test_black_tophat_white_pixel(self):
        for s in self.selems:
            tophat = greyscale_black_top_hat(self.white_pixel, s)
            assert np.all(tophat == 0)

