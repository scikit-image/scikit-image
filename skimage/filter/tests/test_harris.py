import numpy as np

from skimage.filter import harris
from skimage import img_as_float


class TestHarris():
    def test_square_image(self):
        im = np.zeros((50, 50)).astype(float)
        im[:25, :25] = 1.
        results = harris(im)
        assert results.any()
        assert len(results) == 1

    def test_noisy_square_image(self):
        im = np.zeros((50, 50)).astype(float)
        im[:25, :25] = 1.
        im = im + np.random.uniform(size=im.shape) * .5
        results = harris(im)
        assert results.any()
        assert len(results) == 1

    def test_squared_dot(self):
        im = np.zeros((50, 50))
        im[4:8, 4:8] = 1
        im = img_as_float(im)
        results = harris(im)
        assert results == np.array([6, 6])
