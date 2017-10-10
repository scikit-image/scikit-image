import os
import os.path
import numpy as np
from numpy.testing import *
from numpy.testing.decorators import skipif

from tempfile import NamedTemporaryFile

from skimage import data_dir
from skimage.io import imread, imsave, use_plugin, reset_plugins

try:
    import imageio as _imageio
except ImportError:
    imageio_available = False
else:
    imageio_available = True


def setup():
    if imageio_available:
        np.random.seed(0)
        use_plugin('imageio')


def teardown():
    reset_plugins()


@skipif(not imageio_available)
def test_imageio_flatten():
    # a color image is flattened
    img = imread(os.path.join(data_dir, 'color.png'), flatten=True)
    assert img.ndim == 2
    assert img.dtype == np.float64
    img = imread(os.path.join(data_dir, 'camera.png'), flatten=True)
    # check that flattening does not occur for an image that is grey already.
    assert np.sctype2char(img.dtype) in np.typecodes['AllInteger']


@skipif(not imageio_available)
def test_imageio_palette():
    img = imread(os.path.join(data_dir, 'palette_color.png'))
    assert img.ndim == 3


@skipif(not imageio_available)
def test_imageio_truncated_jpg():
    # imageio>2.0 uses Pillow / PIL to try and load the file.
    # Oddly, PIL explicitly raises a SyntaxError when the file read fails.
    assert_raises((RuntimeError, ValueError, SyntaxError),
                  imread,
                  os.path.join(data_dir, 'truncated.jpg'))


class TestSave:

    def roundtrip(self, x, scaling=1):
        f = NamedTemporaryFile(suffix='.png')
        fname = f.name
        f.close()
        imsave(fname, x)
        y = imread(fname)

        assert_array_almost_equal((x * scaling).astype(np.int32), y)

    @skipif(not imageio_available)
    def test_imsave_roundtrip(self):
        dtype = np.uint8
        for shape in [(10, 10), (10, 10, 3), (10, 10, 4)]:
            x = np.ones(shape, dtype=dtype) * np.random.rand(*shape)

            if np.issubdtype(dtype, float):
                yield self.roundtrip, x, 255
            else:
                x = (x * 255).astype(dtype)
                yield self.roundtrip, x

if __name__ == "__main__":
    run_module_suite()
