import os
from tempfile import NamedTemporaryFile

import numpy as np
from skimage import data_dir
from skimage.io import imread, imsave, use_plugin, reset_plugins

from skimage._shared import testing
from skimage._shared.testing import assert_array_almost_equal, TestCase
from skimage._shared._warnings import expected_warnings


def setup():
    use_plugin('imageio')


def teardown():
    reset_plugins()


def test_imageio_as_gray():
    img = imread(os.path.join(data_dir, 'color.png'), as_gray=True)
    assert img.ndim == 2
    assert img.dtype == np.float64
    img = imread(os.path.join(data_dir, 'camera.png'), as_gray=True)
    # check that conversion does not happen for a gray image
    assert np.sctype2char(img.dtype) in np.typecodes['AllInteger']


def test_imageio_palette():
    img = imread(os.path.join(data_dir, 'palette_color.png'))
    assert img.ndim == 3


def test_imageio_truncated_jpg():
    # imageio>2.0 uses Pillow / PIL to try and load the file.
    # Oddly, PIL explicitly raises a SyntaxError when the file read fails.
    with testing.raises(SyntaxError):
        imread(os.path.join(data_dir, 'truncated.jpg'))


class TestSave(TestCase):

    def roundtrip(self, x, scaling=1):
        f = NamedTemporaryFile(suffix='.png')
        fname = f.name
        f.close()
        imsave(fname, x)
        y = imread(fname)

        assert_array_almost_equal((x * scaling).astype(np.int32), y)

    def test_imsave_roundtrip(self):
        dtype = np.uint8
        np.random.seed(0)
        for shape in [(10, 10), (10, 10, 3), (10, 10, 4)]:
            x = np.ones(shape, dtype=dtype) * np.random.rand(*shape)

            if np.issubdtype(dtype, np.floating):
                yield self.roundtrip, x, 255
            else:
                x = (x * 255).astype(dtype)
                yield self.roundtrip, x


def test_return_class():
    testing.assert_equal(
        type(imread(os.path.join(data_dir, 'color.png'))),
        np.ndarray
    )
