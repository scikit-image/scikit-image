from skimage.io._plugins.util import prepare_for_display, WindowManager

from numpy.testing import *
import numpy as np


class TestPrepareForDisplay:
    def test_basic(self):
        prepare_for_display(np.random.random((10, 10)))

    def test_dtype(self):
        x = prepare_for_display(np.random.random((10, 15)))
        assert x.dtype == np.dtype(np.uint8)

    def test_grey(self):
        x = prepare_for_display(np.arange(12, dtype=float).reshape((4, 3)) / 11)
        assert_array_equal(x[..., 0], x[..., 2])
        assert x[0, 0, 0] == 0
        assert x[3, 2, 0] == 255

    def test_colour(self):
        prepare_for_display(np.random.random((10, 10, 3)))

    def test_alpha(self):
        prepare_for_display(np.random.random((10, 10, 4)))

    @raises(ValueError)
    def test_wrong_dimensionality(self):
        prepare_for_display(np.random.random((10, 10, 1, 1)))

    @raises(ValueError)
    def test_wrong_depth(self):
        prepare_for_display(np.random.random((10, 10, 5)))


class TestWindowManager:
    callback_called = False

    def setup(self):
        self.wm = WindowManager()
        self.wm.acquire('test')

    def test_add_window(self):
        self.wm.add_window('window1')
        self.wm.remove_window('window1')

    def callback(self):
        self.callback_called = True

    def test_callback(self):
        self.wm.register_callback(self.callback)
        self.wm.add_window('window')
        self.wm.remove_window('window')
        assert self.callback_called

    def test_has_images(self):
        assert not self.wm.has_windows()
        self.wm.add_window('window')
        assert self.wm.has_windows()

    def teardown(self):
        self.wm._release('test')

if __name__ == "__main__":
    run_module_suite()
