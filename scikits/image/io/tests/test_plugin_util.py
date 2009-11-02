from scikits.image.io._plugins.util import prepare_for_display, WindowManager

from numpy.testing import *
import numpy as np

class TestPrepareForDisplay:
    def test_basic(self):
        prepare_for_display(np.random.random((10, 10)))

    def test_dtype(self):
        x = prepare_for_display(np.random.random((10, 15)))
        assert x.dtype == np.dtype(np.uint8)

    def test_grey(self):
        x = prepare_for_display(np.arange(12, dtype=float).reshape((4,3))/11.)
        assert_array_equal(x[..., 0], x[..., 2])
        assert x[0, 0, 0] == 0
        assert x[3, 2, 0] == 255

    def test_colour(self):
        x = prepare_for_display(np.random.random((10, 10, 3)))

    def test_alpha(self):
        x = prepare_for_display(np.random.random((10, 10, 4)))

    @raises(ValueError)
    def test_wrong_dimensionality(self):
        x = prepare_for_display(np.random.random((10, 10, 1, 1)))

    @raises(ValueError)
    def test_wrong_depth(self):
        x = prepare_for_display(np.random.random((10, 10, 5)))

if __name__ == "__main__":
    run_module_suite()
