import os.path

import numpy as np
from numpy import testing
from scipy import ndimage

import skimage
from skimage import data_dir
from skimage.util import img_as_bool
from skimage.morphology import grey, selem


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
    new_image = grey.white_tophat(image)
    footprint = ndimage.generate_binary_structure(3,1)
    image_expected = ndimage.white_tophat(image,footprint=footprint)
    testing.assert_array_equal(new_image, image_expected)

def test_3d_fallback_black_tophat():
    image = np.ones((7, 7, 7), dtype=bool)
    image[2, 2:4, 2:4] = 0
    image[3, 2:5, 2:5] = 0
    image[4, 3:5, 3:5] = 0
    new_image = grey.black_tophat(image)
    footprint = ndimage.generate_binary_structure(3,1)
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

class TestDTypes():

    def setUp(self):
        k = 5
        arrname = '%03i' % k

        self.disk = selem.disk(k)

        fname_opening = os.path.join(data_dir, "disk-open-matlab-output.npz")
        self.expected_opening = np.load(fname_opening)[arrname]

        fname_closing = os.path.join(data_dir, "disk-close-matlab-output.npz")
        self.expected_closing = np.load(fname_closing)[arrname]

    def _test_image(self, image):
        result_opening = grey.opening(image, self.disk)
        testing.assert_equal(result_opening, self.expected_opening)

        result_closing = grey.closing(image, self.disk)
        testing.assert_equal(result_closing, self.expected_closing)

    def test_float(self):
        image = skimage.img_as_float(lena)
        self._test_image(image)

    @testing.decorators.skipif(True)
    def test_int(self):
        image = skimage.img_as_int(lena)
        self._test_image(image)

    def test_uint(self):
        image = skimage.img_as_uint(lena)
        self._test_image(image)


def test_inplace():
    selem = np.ones((3, 3))
    image = np.zeros((5, 5))
    out = image

    for f in (grey.erosion, grey.dilation,
              grey.white_tophat, grey.black_tophat):
        testing.assert_raises(NotImplementedError, f, image, selem, out=out)


if __name__ == '__main__':
    testing.run_module_suite()
