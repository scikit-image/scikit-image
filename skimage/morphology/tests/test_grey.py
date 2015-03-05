import os.path

import numpy as np
from numpy import testing
from scipy import ndimage

import skimage
from skimage import data_dir, img_as_uint
from skimage.morphology import grey, selem
from skimage._shared._warnings import expected_warnings


lena = np.load(os.path.join(data_dir, 'lena_GRAY_U8.npy'))
bw_lena = lena > 100


class TestMorphology():

    def morph_worker(self, img, fn, morph_func, strel_func):
        matlab_results = np.load(os.path.join(data_dir, fn))
        k = 0
        for arrname in sorted(matlab_results):
            expected_result = matlab_results[arrname]
            mask = strel_func(k)
            actual_result = morph_func(lena, mask)
            testing.assert_equal(expected_result, actual_result)
            k = k + 1

    def test_erode_diamond(self):
        self.morph_worker(lena, "diamond-erode-matlab-output.npz",
                          grey.erosion, selem.diamond)

    def test_dilate_diamond(self):
        self.morph_worker(lena, "diamond-dilate-matlab-output.npz",
                          grey.dilation, selem.diamond)

    def test_open_diamond(self):
        self.morph_worker(lena, "diamond-open-matlab-output.npz",
                          grey.opening, selem.diamond)

    def test_close_diamond(self):
        self.morph_worker(lena, "diamond-close-matlab-output.npz",
                          grey.closing, selem.diamond)

    def test_tophat_diamond(self):
        self.morph_worker(lena, "diamond-tophat-matlab-output.npz",
                          grey.white_tophat, selem.diamond)

    def test_bothat_diamond(self):
        self.morph_worker(lena, "diamond-bothat-matlab-output.npz",
                          grey.black_tophat, selem.diamond)

    def test_erode_disk(self):
        self.morph_worker(lena, "disk-erode-matlab-output.npz",
                          grey.erosion, selem.disk)

    def test_dilate_disk(self):
        self.morph_worker(lena, "disk-dilate-matlab-output.npz",
                          grey.dilation, selem.disk)

    def test_open_disk(self):
        self.morph_worker(lena, "disk-open-matlab-output.npz",
                          grey.opening, selem.disk)

    def test_close_disk(self):
        self.morph_worker(lena, "disk-close-matlab-output.npz",
                          grey.closing, selem.disk)


class TestEccentricStructuringElements():

    def setUp(self):
        self.black_pixel = 255 * np.ones((4, 4), dtype=np.uint8)
        self.black_pixel[1, 1] = 0
        self.white_pixel = 255 - self.black_pixel
        self.selems = [selem.square(2), selem.rectangle(2, 2),
                       selem.rectangle(2, 1), selem.rectangle(1, 2)]

    def test_dilate_erode_symmetry(self):
        for s in self.selems:
            c = grey.erosion(self.black_pixel, s)
            d = grey.dilation(self.white_pixel, s)
            assert np.all(c == (255 - d))

    def test_open_black_pixel(self):
        for s in self.selems:
            grey_open = grey.opening(self.black_pixel, s)
            assert np.all(grey_open == self.black_pixel)

    def test_close_white_pixel(self):
        for s in self.selems:
            grey_close = grey.closing(self.white_pixel, s)
            assert np.all(grey_close == self.white_pixel)

    def test_open_white_pixel(self):
        for s in self.selems:
            assert np.all(grey.opening(self.white_pixel, s) == 0)

    def test_close_black_pixel(self):
        for s in self.selems:
            assert np.all(grey.closing(self.black_pixel, s) == 255)

    def test_white_tophat_white_pixel(self):
        for s in self.selems:
            tophat = grey.white_tophat(self.white_pixel, s)
            assert np.all(tophat == self.white_pixel)

    def test_black_tophat_black_pixel(self):
        for s in self.selems:
            tophat = grey.black_tophat(self.black_pixel, s)
            assert np.all(tophat == (255 - self.black_pixel))

    def test_white_tophat_black_pixel(self):
        for s in self.selems:
            tophat = grey.white_tophat(self.black_pixel, s)
            assert np.all(tophat == 0)

    def test_black_tophat_white_pixel(self):
        for s in self.selems:
            tophat = grey.black_tophat(self.white_pixel, s)
            assert np.all(tophat == 0)

def test_default_selem():
    functions = [grey.erosion, grey.dilation,
                 grey.opening, grey.closing,
                 grey.white_tophat, grey.black_tophat]
    strel = selem.diamond(radius=1)
    image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.uint8)
    for function in functions:
        im_expected = function(image, strel)
        im_test = function(image)
        yield testing.assert_array_equal, im_expected, im_test

def test_3d_fallback_default_selem():
    # 3x3x3 cube inside a 7x7x7 image:
    image = np.zeros((7, 7, 7), np.bool)
    image[2:-2, 2:-2, 2:-2] = 1

    opened = grey.opening(image)

    # expect a "hyper-cross" centered in the 5x5x5:
    image_expected = np.zeros((7, 7, 7), dtype=bool)
    image_expected[2:5, 2:5, 2:5] = ndimage.generate_binary_structure(3, 1)
    testing.assert_array_equal(opened, image_expected)

def test_3d_fallback_cube_selem():
    # 3x3x3 cube inside a 7x7x7 image:
    image = np.zeros((7, 7, 7), np.bool)
    image[2:-2, 2:-2, 2:-2] = 1

    cube = np.ones((3, 3, 3), dtype=np.uint8)

    for function in [grey.closing, grey.opening]:
        new_image = function(image, cube)
        yield testing.assert_array_equal, new_image, image

def test_3d_fallback_white_tophat():
    image = np.zeros((7, 7, 7), dtype=bool)
    image[2, 2:4, 2:4] = 1
    image[3, 2:5, 2:5] = 1
    image[4, 3:5, 3:5] = 1

    with expected_warnings(['operator.*deprecated|\A\Z']):
        new_image = grey.white_tophat(image)
    footprint = ndimage.generate_binary_structure(3,1)
    with expected_warnings(['operator.*deprecated|\A\Z']):
        image_expected = ndimage.white_tophat(image,footprint=footprint)
    testing.assert_array_equal(new_image, image_expected)

def test_3d_fallback_black_tophat():
    image = np.ones((7, 7, 7), dtype=bool)
    image[2, 2:4, 2:4] = 0
    image[3, 2:5, 2:5] = 0
    image[4, 3:5, 3:5] = 0

    with expected_warnings(['operator.*deprecated|\A\Z']):
        new_image = grey.black_tophat(image)
    footprint = ndimage.generate_binary_structure(3,1)
    with expected_warnings(['operator.*deprecated|\A\Z']):
        image_expected = ndimage.black_tophat(image,footprint=footprint)
    testing.assert_array_equal(new_image, image_expected)

def test_2d_ndimage_equivalence():
    image = np.zeros((9, 9), np.uint8)
    image[2:-2, 2:-2] = 128
    image[3:-3, 3:-3] = 196
    image[4, 4] = 255

    opened = grey.opening(image)
    closed = grey.closing(image)

    selem = ndimage.generate_binary_structure(2, 1)
    ndimage_opened = ndimage.grey_opening(image, footprint=selem)
    ndimage_closed = ndimage.grey_closing(image, footprint=selem)

    testing.assert_array_equal(opened, ndimage_opened)
    testing.assert_array_equal(closed, ndimage_closed)

# float test images
im = np.array([[ 0.55,  0.72,  0.6 ,  0.54,  0.42],
               [ 0.65,  0.44,  0.89,  0.96,  0.38],
               [ 0.79,  0.53,  0.57,  0.93,  0.07],
               [ 0.09,  0.02,  0.83,  0.78,  0.87],
               [ 0.98,  0.8 ,  0.46,  0.78,  0.12]])

eroded = np.array([[ 0.55,  0.44,  0.54,  0.42,  0.38],
                   [ 0.44,  0.44,  0.44,  0.38,  0.07],
                   [ 0.09,  0.02,  0.53,  0.07,  0.07],
                   [ 0.02,  0.02,  0.02,  0.78,  0.07],
                   [ 0.09,  0.02,  0.46,  0.12,  0.12]])

dilated = np.array([[ 0.72,  0.72,  0.89,  0.96,  0.54],
                    [ 0.79,  0.89,  0.96,  0.96,  0.96],
                    [ 0.79,  0.79,  0.93,  0.96,  0.93],
                    [ 0.98,  0.83,  0.83,  0.93,  0.87],
                    [ 0.98,  0.98,  0.83,  0.78,  0.87]])

opened = np.array([[ 0.55,  0.55,  0.54,  0.54,  0.42],
                   [ 0.55,  0.44,  0.54,  0.44,  0.38],
                   [ 0.44,  0.53,  0.53,  0.78,  0.07],
                   [ 0.09,  0.02,  0.78,  0.78,  0.78],
                   [ 0.09,  0.46,  0.46,  0.78,  0.12]])

closed = np.array([[ 0.72,  0.72,  0.72,  0.54,  0.54],
                   [ 0.72,  0.72,  0.89,  0.96,  0.54],
                   [ 0.79,  0.79,  0.79,  0.93,  0.87],
                   [ 0.79,  0.79,  0.83,  0.78,  0.87],
                   [ 0.98,  0.83,  0.78,  0.78,  0.78]])

def test_float():
    np.testing.assert_allclose(grey.erosion(im), eroded)
    np.testing.assert_allclose(grey.dilation(im), dilated)
    np.testing.assert_allclose(grey.opening(im), opened)
    np.testing.assert_allclose(grey.closing(im), closed)


def test_uint16():
    im16, eroded16, dilated16, opened16, closed16 = (
                    map(img_as_uint, [im, eroded, dilated, opened, closed]))
    np.testing.assert_allclose(grey.erosion(im16), eroded16)
    np.testing.assert_allclose(grey.dilation(im16), dilated16)
    np.testing.assert_allclose(grey.opening(im16), opened16)
    np.testing.assert_allclose(grey.closing(im16), closed16)


def test_discontiguous_out_array():
    image = np.array([[5, 6, 2],
                      [7, 2, 2],
                      [3, 5, 1]], np.uint8)
    out_array_big = np.zeros((5, 5), np.uint8)
    out_array = out_array_big[::2, ::2]
    expected_dilation = np.array([[7, 0, 6, 0, 6],
                                  [0, 0, 0, 0, 0],
                                  [7, 0, 7, 0, 2],
                                  [0, 0, 0, 0, 0],
                                  [7, 0, 5, 0, 5]], np.uint8)
    expected_erosion = np.array([[5, 0, 2, 0, 2],
                                 [0, 0, 0, 0, 0],
                                 [2, 0, 2, 0, 1],
                                 [0, 0, 0, 0, 0],
                                 [3, 0, 1, 0, 1]], np.uint8)
    grey.dilation(image, out=out_array)
    testing.assert_array_equal(out_array_big, expected_dilation)
    grey.erosion(image, out=out_array)
    testing.assert_array_equal(out_array_big, expected_erosion)


if __name__ == '__main__':
    testing.run_module_suite()
