import numpy as np
import skimage.data as data
from numpy.testing import assert_equal, assert_almost_equal, assert_raises


def test_lena_removed():
    """ Test that "Lena" has been removed """
    assert_raises(RuntimeError, data.lena)


def test_astronaut():
    """ Test that "astronaut" image can be loaded. """
    astronaut = data.astronaut()
    assert_equal(astronaut.shape, (512, 512, 3))


def test_camera():
    """ Test that "camera" image can be loaded. """
    cameraman = data.camera()
    assert_equal(cameraman.ndim, 2)


def test_checkerboard():
    """ Test that "checkerboard" image can be loaded. """
    data.checkerboard()


def test_text():
    """ Test that "text" image can be loaded. """
    data.text()


def test_moon():
    """ Test that "moon" image can be loaded. """
    data.moon()


def test_page():
    """ Test that "page" image can be loaded. """
    data.page()


def test_clock():
    """ Test that "clock" image can be loaded. """
    data.clock()


def test_chelsea():
    """ Test that "chelsea" image can be loaded. """
    data.chelsea()


def test_coffee():
    """ Test that "coffee" image can be loaded. """
    data.coffee()


def test_binary_blobs():
    blobs = data.binary_blobs(length=128)
    assert_almost_equal(blobs.mean(), 0.5, decimal=1)
    blobs = data.binary_blobs(length=128, volume_fraction=0.25)
    assert_almost_equal(blobs.mean(), 0.25, decimal=1)
    blobs = data.binary_blobs(length=32, volume_fraction=0.25, n_dim=3)
    assert_almost_equal(blobs.mean(), 0.25, decimal=1)
    other_realization = data.binary_blobs(length=32, volume_fraction=0.25,
                                          n_dim=3)
    assert not np.all(blobs == other_realization)


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
