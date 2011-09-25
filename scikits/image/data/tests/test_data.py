import unittest

import scikits.image.data as data

class TestData(unittest.TestCase):
    def test_lena(self):
        """ Test that "Lena" image can be loaded. """
        data.lena()

    def test_camera(self):
        """ Test that "camera" image can be loaded. """
        data.camera()

    def test_checkerboard(self):
        """ Test that checkerboard image can be loaded. """
        data.checkerboard()

    def test_checkerboard_gray(self):
        """ Test that checkerboard grayscale image can be loaded. """
        data.checkerboard_gray()


