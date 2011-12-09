import numpy as np

import skimage
from skimage import data
from skimage.thresholding import otsu_threshold, binarize


class TestSimpleImage():
    def setup(self):
        self.image = np.array([[0, 0, 1, 3, 5],
                               [0, 1, 4, 3, 4],
                               [1, 2, 5, 4, 1],
                               [2, 4, 5, 2, 1],
                               [4, 5, 1, 0, 0]], dtype=int)

    def test_otsu(self):
        assert otsu_threshold(self.image) == 2

    @np.testing.raises(NotImplementedError)
    def test_otsu_raises_error(self):
        image = self.image - 2
        otsu_threshold(image)

    def test_otsu_float_image(self):
        image = np.float64(self.image)
        assert 2 <= otsu_threshold(image) < 3

    def test_binarize(self):
        expected = np.array([[0, 0, 0, 1, 1],
                             [0, 0, 1, 1, 1],
                             [0, 0, 1, 1, 0],
                             [0, 1, 1, 0, 0],
                             [1, 1, 0, 0, 0]])
        assert np.all(binarize(self.image) == expected)


def test_otsu_camera_image():
    assert otsu_threshold(data.camera()) == 87

def test_otsu_coins_image():
    assert otsu_threshold(data.coins()) == 107

def test_otsu_coins_image_as_float():
    coins = skimage.img_as_float(data.coins())
    assert 0.41 < otsu_threshold(coins) < 0.42

def test_otsu_lena_image():
    assert otsu_threshold(data.lena()) == 141


if __name__ == '__main__':
    np.testing.run_module_suite()

