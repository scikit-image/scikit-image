import numpy as np
from numpy.testing import assert_array_equal

import skimage
from skimage import data
from skimage.filter.thresholding import (threshold_adaptive,
                                         threshold_otsu,
                                         threshold_yen)


class TestSimpleImage():
    def setup(self):
        self.image = np.array([[0, 0, 1, 3, 5],
                               [0, 1, 4, 3, 4],
                               [1, 2, 5, 4, 1],
                               [2, 4, 5, 2, 1],
                               [4, 5, 1, 0, 0]], dtype=int)

    def test_otsu(self):
        assert threshold_otsu(self.image) == 2

    def test_otsu_negative_int(self):
        image = self.image - 2
        assert threshold_otsu(image) == 0

    def test_otsu_float_image(self):
        image = np.float64(self.image)
        assert 2 <= threshold_otsu(image) < 3

    def test_yen(self):
        assert threshold_yen(self.image) == 2

    def test_yen_negative_int(self):
        image = self.image - 2
        assert threshold_yen(image) == 0

    def test_yen_float_image(self):
        image = np.float64(self.image)
        assert 2 <= threshold_yen(image) < 3

    def test_yen_arange(self):
        image = np.arange(256)
        assert threshold_yen(image) == 127

    def test_yen_binary(self):
        image = np.zeros([2,256], dtype='uint8')
        image[0] = 255
        assert threshold_yen(image) < 1

    def test_threshold_adaptive_generic(self):
        def func(arr):
            return arr.sum() / arr.shape[0]
        ref = np.array(
            [[False, False, False, False,  True],
             [False, False,  True, False,  True],
             [False, False,  True,  True, False],
             [False,  True,  True, False, False],
             [ True,  True, False, False, False]]
        )
        out = threshold_adaptive(self.image, 3, method='generic', param=func)
        assert_array_equal(ref, out)

    def test_threshold_adaptive_gaussian(self):
        ref = np.array(
            [[False, False, False, False,  True],
             [False, False,  True, False,  True],
             [False, False,  True,  True, False],
             [False,  True,  True, False, False],
             [ True,  True, False, False, False]]
        )
        out = threshold_adaptive(self.image, 3, method='gaussian')
        assert_array_equal(ref, out)

    def test_threshold_adaptive_mean(self):
        ref = np.array(
            [[False, False, False, False,  True],
             [False, False,  True, False,  True],
             [False, False,  True,  True, False],
             [False,  True,  True, False, False],
             [ True,  True, False, False, False]]
        )
        out = threshold_adaptive(self.image, 3, method='mean')
        assert_array_equal(ref, out)

    def test_threshold_adaptive_median(self):
        ref = np.array(
            [[False, False, False, False,  True],
             [False, False,  True, False, False],
             [False, False,  True, False, False],
             [False, False,  True,  True, False],
             [False,  True, False, False, False]]
        )
        out = threshold_adaptive(self.image, 3, method='median')
        assert_array_equal(ref, out)


def test_otsu_camera_image():
    camera = skimage.img_as_ubyte(data.camera())
    assert 86 < threshold_otsu(camera) < 88


def test_otsu_coins_image():
    coins = skimage.img_as_ubyte(data.coins())
    assert 106 < threshold_otsu(coins) < 108


def test_otsu_coins_image_as_float():
    coins = skimage.img_as_float(data.coins())
    assert 0.41 < threshold_otsu(coins) < 0.42


def test_otsu_lena_image():
    lena = skimage.img_as_ubyte(data.lena())
    assert 140 < threshold_otsu(lena) < 142


def test_yen_coins_image():
    coins = skimage.img_as_ubyte(data.coins())
    assert 109 < threshold_yen(coins) < 111


def test_yen_coins_image_as_float():
    coins = skimage.img_as_float(data.coins())
    assert 0.43 < threshold_yen(coins) < 0.44


if __name__ == '__main__':
    np.testing.run_module_suite()
