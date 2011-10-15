import os.path

import numpy as np
from numpy.testing import *

from scikits.image import data_dir
from scikits.image.io import imread
from scikits.image import data_dir
from scikits.image.morphology import *

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


BLACK_PIXEL = 255 * np.ones((4, 4), dtype=np.uint8)
BLACK_PIXEL[1, 1] = 0
WHITE_PIXEL = 255 - BLACK_PIXEL
SELEM = square(2)

def test_dilate_erode_symmetry():
    c = greyscale_erode(BLACK_PIXEL, SELEM)
    d = greyscale_dilate(WHITE_PIXEL, SELEM)
    assert np.all(c == (255 - d))


def test_open_dark_pixel():
    assert np.all(greyscale_open(BLACK_PIXEL, SELEM) == BLACK_PIXEL)

def test_close_white_pixel():
    assert np.all(greyscale_close(WHITE_PIXEL, SELEM) == WHITE_PIXEL)


def test_open_white_pixel():
    assert np.all(greyscale_open(WHITE_PIXEL, SELEM) == 0)

def test_close_dark_pixel():
    assert np.all(greyscale_close(BLACK_PIXEL, SELEM) == 255)


def test_white_tophat_white_pixel():
    tophat = greyscale_white_top_hat(WHITE_PIXEL, SELEM)
    assert np.all(tophat == WHITE_PIXEL)

def test_black_tophat_black_pixel():
    tophat = greyscale_black_top_hat(BLACK_PIXEL, SELEM)
    assert np.all(tophat == (255 - BLACK_PIXEL))


def test_white_tophat_black_pixel():
    tophat = greyscale_white_top_hat(BLACK_PIXEL, SELEM)
    assert np.all(tophat == 0)

def test_black_tophat_white_pixel():
    tophat = greyscale_black_top_hat(WHITE_PIXEL, SELEM)
    assert np.all(tophat == 0)


