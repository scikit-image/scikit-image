import os.path
import numpy as np
from numpy.testing import *
from numpy.testing.decorators import skipif

from tempfile import NamedTemporaryFile

from skimage import data_dir
from skimage.io import imread, imsave, use_plugin, reset_plugins

try:
    from PIL import Image
    from skimage.io._plugins.pil_plugin import _palette_is_grayscale
    use_plugin('pil')
except ImportError:
    PIL_available = False
else:
    PIL_available = True


def teardown():
    reset_plugins()


def setup_module(self):
    """The effect of the `plugin.use` call may be overridden by later imports.
    Call `use_plugin` directly before the tests to ensure that PIL is used.

    """
    try:
        use_plugin('pil')
    except ImportError:
        pass



@skipif(not PIL_available)
def test_imread_flatten():
    # a color image is flattened
    img = imread(os.path.join(data_dir, 'color.png'), flatten=True)
    assert img.ndim == 2
    assert img.dtype == np.float64
    img = imread(os.path.join(data_dir, 'camera.png'), flatten=True)
    # check that flattening does not occur for an image that is grey already.
    assert np.sctype2char(img.dtype) in np.typecodes['AllInteger']


@skipif(not PIL_available)
def test_imread_palette():
    img = imread(os.path.join(data_dir, 'palette_gray.png'))
    assert img.ndim == 2
    img = imread(os.path.join(data_dir, 'palette_color.png'))
    assert img.ndim == 3


@skipif(not PIL_available)
def test_palette_is_gray():
    from PIL import Image
    gray = Image.open(os.path.join(data_dir, 'palette_gray.png'))
    assert _palette_is_grayscale(gray)
    color = Image.open(os.path.join(data_dir, 'palette_color.png'))
    assert not _palette_is_grayscale(color)


@skipif(not PIL_available)
def test_bilevel():
    expected = np.zeros((10, 10))
    expected[::2] = 255

    img = imread(os.path.join(data_dir, 'checker_bilevel.png'))
    assert_array_equal(img, expected)


@skipif(not PIL_available)
def test_imread_uint16():
    expected = np.load(os.path.join(data_dir, 'chessboard_GRAY_U8.npy'))
    img = imread(os.path.join(data_dir, 'chessboard_GRAY_U16.tif'))
    assert np.issubdtype(img.dtype, np.uint16)
    assert_array_almost_equal(img, expected)


@skipif(not PIL_available)
def test_repr_png():
    img_path = os.path.join(data_dir, 'camera.png')
    original_img = imread(img_path)
    original_img_str = original_img._repr_png_()

    with NamedTemporaryFile(suffix='.png') as temp_png:
        temp_png.write(original_img_str)
        temp_png.seek(0)
        round_trip = imread(temp_png)

    assert np.all(original_img == round_trip)


# Big endian images not correctly loaded for PIL < 1.1.7
# Renable test when PIL 1.1.7 is more common.
@skipif(True)
def test_imread_uint16_big_endian():
    expected = np.load(os.path.join(data_dir, 'chessboard_GRAY_U8.npy'))
    img = imread(os.path.join(data_dir, 'chessboard_GRAY_U16B.tif'))
    assert img.dtype == np.dtype('>u2')
    assert_array_almost_equal(img, expected)


class TestSave:
    def roundtrip(self, dtype, x, scaling=1):
        f = NamedTemporaryFile(suffix='.png')
        fname = f.name
        f.close()
        imsave(fname, x)
        y = imread(fname)

        assert_array_almost_equal((x * scaling).astype(np.int32), y)

    @skipif(not PIL_available)
    def test_imsave_roundtrip(self):
        for shape in [(10, 10), (10, 10, 3), (10, 10, 4)]:
            for dtype in (np.uint8, np.uint16, np.float32, np.float64):
                x = np.ones(shape, dtype=dtype) * np.random.random(shape)

                if np.issubdtype(dtype, float):
                    yield self.roundtrip, dtype, x, 255
                else:
                    x = (x * 255).astype(dtype)
                    yield self.roundtrip, dtype, x

if __name__ == "__main__":
    run_module_suite()
