import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

import skimage
from skimage import data
from skimage.filters.thresholding import (threshold_adaptive,
                                          threshold_otsu,
                                          threshold_li,
                                          threshold_yen,
                                          threshold_isodata)


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

    def test_li(self):
        assert int(threshold_li(self.image)) == 2

    def test_li_negative_int(self):
        image = self.image - 2
        assert int(threshold_li(image)) == 0

    def test_li_float_image(self):
        image = np.float64(self.image)
        assert 2 <= threshold_li(image) < 3

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
        image = np.zeros([2,256], dtype=np.uint8)
        image[0] = 255
        assert threshold_yen(image) < 1

    def test_yen_blank_zero(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        assert threshold_yen(image) == 0

    def test_yen_blank_max(self):
        image = np.empty((5, 5), dtype=np.uint8)
        image.fill(255)
        assert threshold_yen(image) == 255

    def test_isodata(self):
        assert threshold_isodata(self.image) == 2
        assert threshold_isodata(self.image, return_all=True) == [2]

    def test_isodata_blank_zero(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        assert threshold_isodata(image) == 0
        assert threshold_isodata(image, return_all=True) == [0]

    def test_isodata_linspace(self):
        image = np.linspace(-127, 0, 256)
        assert -63.8 < threshold_isodata(image) < -63.6
        assert_almost_equal(threshold_isodata(image, return_all=True),
                            [-63.74804688, -63.25195312])

    def test_isodata_16bit(self):
        np.random.seed(0)
        imfloat = np.random.rand(256, 256)
        assert 0.49 < threshold_isodata(imfloat, nbins=1024) < 0.51
        assert all(0.49 < threshold_isodata(imfloat, nbins=1024,
                                            return_all=True))

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
        assert_equal(ref, out)

    def test_threshold_adaptive_gaussian(self):
        ref = np.array(
            [[False, False, False, False,  True],
             [False, False,  True, False,  True],
             [False, False,  True,  True, False],
             [False,  True,  True, False, False],
             [ True,  True, False, False, False]]
        )
        out = threshold_adaptive(self.image, 3, method='gaussian')
        assert_equal(ref, out)

        out = threshold_adaptive(self.image, 3, method='gaussian', param=1.0 / 3.0)
        assert_equal(ref, out)

    def test_threshold_adaptive_mean(self):
        ref = np.array(
            [[False, False, False, False,  True],
             [False, False,  True, False,  True],
             [False, False,  True,  True, False],
             [False,  True,  True, False, False],
             [ True,  True, False, False, False]]
        )
        out = threshold_adaptive(self.image, 3, method='mean')
        assert_equal(ref, out)

    def test_threshold_adaptive_median(self):
        ref = np.array(
            [[False, False, False, False,  True],
             [False, False,  True, False, False],
             [False, False,  True, False, False],
             [False, False,  True,  True, False],
             [False,  True, False, False, False]]
        )
        out = threshold_adaptive(self.image, 3, method='median')
        assert_equal(ref, out)


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
    img = skimage.img_as_ubyte(data.lena())
    assert 140 < threshold_otsu(img) < 142

def test_otsu_astro_image():
    img = skimage.img_as_ubyte(data.astronaut())
    assert 109 < threshold_otsu(img) < 111

def test_li_camera_image():
    camera = skimage.img_as_ubyte(data.camera())
    assert 63 < threshold_li(camera) < 65


def test_li_coins_image():
    coins = skimage.img_as_ubyte(data.coins())
    assert 95 < threshold_li(coins) < 97


def test_li_coins_image_as_float():
    coins = skimage.img_as_float(data.coins())
    assert 0.37 < threshold_li(coins) < 0.38


def test_li_astro_image():
    img = skimage.img_as_ubyte(data.astronaut())
    assert 66 < threshold_li(img) < 68

def test_yen_camera_image():
    camera = skimage.img_as_ubyte(data.camera())
    assert 197 < threshold_yen(camera) < 199


def test_yen_coins_image():
    coins = skimage.img_as_ubyte(data.coins())
    assert 109 < threshold_yen(coins) < 111


def test_yen_coins_image_as_float():
    coins = skimage.img_as_float(data.coins())
    assert 0.43 < threshold_yen(coins) < 0.44


def test_isodata_camera_image():
    camera = skimage.img_as_ubyte(data.camera())

    threshold = threshold_isodata(camera)
    assert np.floor((camera[camera <= threshold].mean() +
                     camera[camera > threshold].mean()) / 2.0) == threshold
    assert threshold == 87

    assert threshold_isodata(camera, return_all=True) == [87]


def test_isodata_coins_image():
    coins = skimage.img_as_ubyte(data.coins())

    threshold = threshold_isodata(coins)
    assert np.floor((coins[coins <= threshold].mean() +
                     coins[coins > threshold].mean()) / 2.0) == threshold
    assert threshold == 107

    assert threshold_isodata(coins, return_all=True) == [107]


def test_isodata_moon_image():
    moon = skimage.img_as_ubyte(data.moon())

    threshold = threshold_isodata(moon)
    assert np.floor((moon[moon <= threshold].mean() +
                     moon[moon > threshold].mean()) / 2.0) == threshold
    assert threshold == 86

    thresholds = threshold_isodata(moon, return_all=True)
    for threshold in thresholds:
        assert np.floor((moon[moon <= threshold].mean() +
                         moon[moon > threshold].mean()) / 2.0) == threshold
    assert_equal(thresholds, [86, 87, 88, 122, 123, 124, 139, 140])


def test_isodata_moon_image_negative_int():
    moon = skimage.img_as_ubyte(data.moon()).astype(np.int32)
    moon -= 100

    threshold = threshold_isodata(moon)
    assert np.floor((moon[moon <= threshold].mean() +
                     moon[moon > threshold].mean()) / 2.0) == threshold
    assert threshold == -14

    thresholds = threshold_isodata(moon, return_all=True)
    for threshold in thresholds:
        assert np.floor((moon[moon <= threshold].mean() +
                         moon[moon > threshold].mean()) / 2.0) == threshold
    assert_equal(thresholds, [-14, -13, -12,  22,  23,  24,  39,  40])


def test_isodata_moon_image_negative_float():
    moon = skimage.img_as_ubyte(data.moon()).astype(np.float64)
    moon -= 100

    assert -14 < threshold_isodata(moon) < -13

    thresholds = threshold_isodata(moon, return_all=True)
    assert_almost_equal(thresholds,
                        [-13.83789062, -12.84179688, -11.84570312, 22.02148438,
                         23.01757812, 24.01367188, 38.95507812, 39.95117188])


if __name__ == '__main__':
    np.testing.run_module_suite()
