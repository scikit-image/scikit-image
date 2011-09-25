import scikits.image.data as data
from numpy.testing import assert_equal, assert_array_equal
import numpy as np

def test_lena():
    """ Test that "Lena" image can be loaded. """
    lena = data.lena()
    assert_equal(lena.shape, (512, 512, 3))

def test_camera():
    """ Test that "camera" image can be loaded. """
    cameraman = data.camera()
    assert_equal(cameraman.ndim, 2)

def test_checkerboard():
    """ Test that checkerboard image can be loaded. """
    checkerboard = data.checkerboard()
    assert_equal(checkerboard.dtype, np.uint8)

if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()

