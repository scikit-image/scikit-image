import os.path
import numpy as np
import unittest

from tempfile import NamedTemporaryFile

from skimage import data
from skimage.io import imread, imsave, use_plugin, reset_plugins
from skimage._shared import testing

from pytest import importorskip

importorskip('SimpleITK')

np.random.seed(0)

def teardown():
    reset_plugins()


def setup_module(self):
    """The effect of the `plugin.use` call may be overridden by later imports.
    Call `use_plugin` directly before the tests to ensure that SimpleITK is
    used.

    """
    use_plugin('simpleitk')


def test_imread_as_gray():
    img = imread(testing.fetch('data/color.png'), as_gray=True)
    assert img.ndim == 2
    assert img.dtype == np.float64
    img = imread(testing.fetch('data/camera.png'), as_gray=True)
    # check that conversion does not happen for a gray image
    assert np.sctype2char(img.dtype) in np.typecodes['AllInteger']


def test_bilevel():
    expected = np.zeros((10, 10))
    expected[::2] = 255

    img = imread(testing.fetch('data/checker_bilevel.png'))
    np.testing.assert_array_equal(img, expected)

"""
#TODO: This test causes a Segmentation fault
def test_imread_truncated_jpg():
    assert_raises((RuntimeError, ValueError),
                  imread,
                  testing.fetch('data/truncated.jpg'))
"""


def test_imread_uint16():
    expected = np.load(testing.fetch('data/chessboard_GRAY_U8.npy'))
    img = imread(testing.fetch('data/chessboard_GRAY_U16.tif'))
    assert np.issubdtype(img.dtype, np.uint16)
    np.testing.assert_array_almost_equal(img, expected)


def test_imread_uint16_big_endian():
    expected = np.load(testing.fetch('data/chessboard_GRAY_U8.npy'))
    img = imread(testing.fetch('data/chessboard_GRAY_U16B.tif'))
    np.testing.assert_array_almost_equal(img, expected)


class TestSave(unittest.TestCase):
    def roundtrip(self, dtype, x):
        f = NamedTemporaryFile(suffix='.mha')
        fname = f.name
        f.close()
        imsave(fname, x)
        y = imread(fname)

        np.testing.assert_array_almost_equal(x, y)

    def test_imsave_roundtrip(self):
        for shape in [(10, 10), (10, 10, 3), (10, 10, 4)]:
            for dtype in (np.uint8, np.uint16, np.float32, np.float64):
                x = np.ones(shape, dtype=dtype) * np.random.rand(*shape)

                if np.issubdtype(dtype, np.floating):
                    yield self.roundtrip, dtype, x
                else:
                    x = (x * 255).astype(dtype)
                    yield self.roundtrip, dtype, x
