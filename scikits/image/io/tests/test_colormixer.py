from numpy.testing import *
import numpy as np

import scikits.image.io._plugins._colormixer as cm

class ColorMixerTest(object):
    def setup(self):
        self.state = np.ones((18, 33, 3), dtype=np.uint8) * 200
        self.img = np.zeros_like(self.state)

    def test_basic(self):
        self.op(self.img, self.state, 0, self.positive)
        assert_array_equal(self.img[..., 0],
                           self.py_op(self.state[..., 0], self.positive))

    def test_clip(self):
        self.op(self.img, self.state, 0, self.positive_clip)
        assert_array_equal(self.img[..., 0],
                           np.ones_like(self.img[..., 0]) * 255)

    def test_negative(self):
        self.op(self.img, self.state, 0, self.negative)
        assert_array_equal(self.img[..., 0],
                           self.py_op(self.state[..., 0], self.negative))

    def test_negative_clip(self):
        self.op(self.img, self.state, 0, self.negative_clip)
        assert_array_equal(self.img[..., 0],
                           np.zeros_like(self.img[..., 0]))

class TestColorMixerAdd(ColorMixerTest):
    op = cm.add
    py_op = np.add
    positive = 50
    positive_clip = 56
    negative = -50
    negative_clip = -220

class TestColorMixerMul(ColorMixerTest):
    op = cm.multiply
    py_op = np.multiply
    positive = 1.2
    positive_clip = 2
    negative = 0.5
    negative_clip = -0.5


if __name__ == "__main__":
    run_module_suite()
