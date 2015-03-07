import os.path
import numpy as np
from numpy.testing import (
    assert_array_equal, assert_array_almost_equal, assert_raises,
    assert_allclose, run_module_suite)

from tempfile import NamedTemporaryFile

from ... import data_dir
from .. import (imread, imsave, use_plugin, reset_plugins,
                        Image as ioImage)
from ..._shared.testing import mono_check, color_check
from ..._shared._warnings import expected_warnings

from six import BytesIO

from PIL import Image
from .._plugins.pil_plugin import (
    pil_to_ndarray, ndarray_to_pil, _palette_is_grayscale)
from ...measure import structural_similarity as ssim
from ...color import rgb2lab


def setup():
    use_plugin('pil')


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


def test_imread_flatten():
    # a color image is flattened
    img = imread(os.path.join(data_dir, 'color.png'), flatten=True)
    assert img.ndim == 2
    assert img.dtype == np.float64
    img = imread(os.path.join(data_dir, 'camera.png'), flatten=True)
    # check that flattening does not occur for an image that is grey already.
    assert np.sctype2char(img.dtype) in np.typecodes['AllInteger']


def test_imread_palette():
    img = imread(os.path.join(data_dir, 'palette_gray.png'))
    assert img.ndim == 2
    img = imread(os.path.join(data_dir, 'palette_color.png'))
    assert img.ndim == 3


def test_palette_is_gray():
    gray = Image.open(os.path.join(data_dir, 'palette_gray.png'))
    assert _palette_is_grayscale(gray)
    color = Image.open(os.path.join(data_dir, 'palette_color.png'))
    assert not _palette_is_grayscale(color)


def test_bilevel():
    expected = np.zeros((10, 10))
    expected[::2] = 255

    img = imread(os.path.join(data_dir, 'checker_bilevel.png'))
    assert_array_equal(img, expected)


def test_imread_uint16():
    expected = np.load(os.path.join(data_dir, 'chessboard_GRAY_U8.npy'))
    img = imread(os.path.join(data_dir, 'chessboard_GRAY_U16.tif'))
    assert np.issubdtype(img.dtype, np.uint16)
    assert_array_almost_equal(img, expected)


def test_repr_png():
    img_path = os.path.join(data_dir, 'camera.png')
    original_img = ioImage(imread(img_path))
    original_img_str = original_img._repr_png_()

    with NamedTemporaryFile(suffix='.png') as temp_png:
        temp_png.write(original_img_str)
        temp_png.seek(0)
        round_trip = imread(temp_png)

    assert np.all(original_img == round_trip)


def test_imread_truncated_jpg():
    assert_raises((IOError, ValueError), imread,
                  os.path.join(data_dir, 'truncated.jpg'))


def test_imread_uint16_big_endian():
    expected = np.load(os.path.join(data_dir, 'chessboard_GRAY_U8.npy'))
    img = imread(os.path.join(data_dir, 'chessboard_GRAY_U16B.tif'))
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)


class TestSave:
    def roundtrip_file(self, x):
        f = NamedTemporaryFile(suffix='.png')
        fname = f.name
        f.close()
        imsave(fname, x)
        y = imread(fname)
        return y

    def roundtrip_pil_image(self, x):
        pil_image = ndarray_to_pil(x)
        y = pil_to_ndarray(pil_image)
        return y

    def verify_roundtrip(self, dtype, x, y, scaling=1):
        assert_array_almost_equal((x * scaling).astype(np.int32), y)

    def verify_imsave_roundtrip(self, roundtrip_function):
        for shape in [(10, 10), (10, 10, 3), (10, 10, 4)]:
            for dtype in (np.uint8, np.uint16, np.float32, np.float64):
                x = np.ones(shape, dtype=dtype) * np.random.rand(*shape)

                if np.issubdtype(dtype, float):
                    yield (self.verify_roundtrip, dtype, x,
                           roundtrip_function(x), 255)
                else:
                    x = (x * 255).astype(dtype)
                    yield (self.verify_roundtrip, dtype, x,
                           roundtrip_function(x))

    def test_imsave_roundtrip_file(self):
        self.verify_imsave_roundtrip(self.roundtrip_file)

    def test_imsave_roundtrip_pil_image(self):
        self.verify_imsave_roundtrip(self.roundtrip_pil_image)


def test_imsave_filelike():
    shape = (2, 2)
    image = np.zeros(shape)
    s = BytesIO()

    # save to file-like object
    with expected_warnings(['precision loss|unclosed file',
                            'is a low contrast image']):
        imsave(s, image)

    # read from file-like object
    s.seek(0)
    out = imread(s)
    assert out.shape == shape
    assert_allclose(out, image)


def test_imexport_imimport():
    shape = (2, 2)
    image = np.zeros(shape)
    with expected_warnings(['precision loss']):
        pil_image = ndarray_to_pil(image)
    out = pil_to_ndarray(pil_image)
    assert out.shape == shape


def test_all_color():
    color_check('pil')
    color_check('pil', 'bmp')


def test_all_mono():
    mono_check('pil')
    mono_check('pil', 'tiff')


def test_multi_page_gif():
    img = imread(os.path.join(data_dir, 'no_time_for_that.gif'))
    assert img.shape == (24, 280, 500, 3), img.shape
    img2 = imread(os.path.join(data_dir, 'no_time_for_that.gif'),
                  img_num=5)
    assert img2.shape == (280, 500, 3)
    assert_allclose(img[5], img2)


def test_cmyk():
    ref = imread(os.path.join(data_dir, 'color.png'))

    img = Image.open(os.path.join(data_dir, 'color.png'))
    img = img.convert('CMYK')

    f = NamedTemporaryFile(suffix='.jpg')
    fname = f.name
    f.close()
    img.save(fname)
    try:
        img.close()
    except AttributeError:  # `close` not available on PIL
        pass

    new = imread(fname)

    ref_lab = rgb2lab(ref)
    new_lab = rgb2lab(new)

    for i in range(3):
        newi = np.ascontiguousarray(new_lab[:, :, i])
        refi = np.ascontiguousarray(ref_lab[:, :, i])
        sim = ssim(refi, newi, dynamic_range=refi.max() - refi.min())
        assert sim > 0.99


class TestSaveTIF:
    def roundtrip(self, dtype, x):
        f = NamedTemporaryFile(suffix='.tif')
        fname = f.name
        f.close()
        imsave(fname, x)
        y = imread(fname)
        assert_array_equal(x, y)

    def test_imsave_roundtrip(self):
        for shape in [(10, 10), (10, 10, 3), (10, 10, 4)]:
            for dtype in (np.uint8, np.uint16, np.int16, np.float32,
                          np.float64, np.bool):
                x = np.random.rand(*shape)

                if not np.issubdtype(dtype, float) and not dtype == np.bool:
                    x = (x * np.iinfo(dtype).max).astype(dtype)
                else:
                    x = x.astype(dtype)
                yield self.roundtrip, dtype, x

if __name__ == "__main__":
    run_module_suite()
