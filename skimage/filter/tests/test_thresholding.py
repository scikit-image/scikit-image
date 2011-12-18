import numpy as np

import skimage
from skimage import data
from skimage.filter.thresholding import threshold_otsu


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


def test_otsu_camera_image():
    assert threshold_otsu(data.camera()) == 87

def test_otsu_coins_image():
    assert threshold_otsu(data.coins()) == 107

def test_otsu_coins_image_as_float():
    coins = skimage.img_as_float(data.coins())
    assert 0.41 < threshold_otsu(coins) < 0.42

def test_otsu_lena_image():
    assert threshold_otsu(data.lena()) == 141


if __name__ == '__main__':
    np.testing.run_module_suite()

