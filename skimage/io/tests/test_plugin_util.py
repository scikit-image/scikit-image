from skimage.io._plugins.util import prepare_for_display, WindowManager
from skimage._shared._warnings import expected_warnings

from numpy.testing import assert_array_equal
import unittest
import pytest
import numpy as np

np.random.seed(0)


class TestPrepareForDisplay(unittest.TestCase):
    def test_basic(self):
        with expected_warnings(['precision loss']):
            prepare_for_display(np.random.rand(10, 10))

    def test_dtype(self):
        with expected_warnings(['precision loss']):
            x = prepare_for_display(np.random.rand(10, 15))
        assert x.dtype == np.dtype(np.uint8)

    def test_grey(self):
        with expected_warnings(['precision loss']):
            tmp = np.arange(12, dtype=float).reshape((4, 3)) / 11
            x = prepare_for_display(tmp)
        assert_array_equal(x[..., 0], x[..., 2])
        assert x[0, 0, 0] == 0
        assert x[3, 2, 0] == 255

    def test_color(self):
        with expected_warnings(['precision loss']):
            prepare_for_display(np.random.rand(10, 10, 3))

    def test_alpha(self):
        with expected_warnings(['precision loss']):
            prepare_for_display(np.random.rand(10, 10, 4))

    def test_wrong_dimensionality(self):
        with pytest.raises(ValueError):
            with expected_warnings(['precision loss']):
                prepare_for_display(np.random.rand(10, 10, 1, 1))

    def test_wrong_depth(self):
        with pytest.raises(ValueError):
            with expected_warnings(['precision loss']):
                prepare_for_display(np.random.rand(10, 10, 5))


class TestWindowManager(unittest.TestCase):
    callback_called = False

    @pytest.fixture(autouse=True)
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
    from numpy.testing import run_module_suite
    run_module_suite()
