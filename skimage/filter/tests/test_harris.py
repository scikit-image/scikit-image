import numpy as np

import unittest

from skimage.filter import harris_corner_detector


class TestHarris(unittest.TestCase):

    def test_square_image(self):
        im = np.zeros((50, 50)).astype(float)
        im[:25, :25] = 1.
        results = harris_corner_detector(im)
        self.assertTrue(results.any())
        self.assertTrue(len(results) == 1)

    def test_noisy_square_image(self):
        im = np.zeros((50, 50)).astype(float)
        im[:25, :25] = 1.
        im = im + np.random.uniform(size=im.shape) * .5
        results = harris_corner_detector(im)
        assert results.any()
        assert len(results) == 1
