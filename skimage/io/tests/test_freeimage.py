import os
import skimage as si
import skimage.io as sio
from skimage import data_dir
import numpy as np

from numpy.testing import *
from numpy.testing.decorators import skipif
from tempfile import NamedTemporaryFile

try:
    import skimage.io._plugins.freeimage_plugin as fi
    FI_available = True
    sio.use_plugin('freeimage')
except RuntimeError:
    FI_available = False

np.random.seed(0)


def setup_module(self):
    """The effect of the `plugin.use` call may be overridden by later imports.
    Call `use_plugin` directly before the tests to ensure that freeimage is
    used.

    """
    try:
        sio.use_plugin('freeimage')
    except RuntimeError:
        pass


def teardown():
    sio.reset_plugins()


@skipif(not FI_available)
def test_imread():
    img = sio.imread(os.path.join(si.data_dir, 'color.png'))
    assert img.shape == (370, 371, 3)
    assert all(img[274, 135] == [0, 130, 253])

@skipif(not FI_available)
def test_imread_truncated_jpg():
    assert_raises((RuntimeError, ValueError),
                  sio.imread,
                  os.path.join(si.data_dir, 'truncated.jpg'))

@skipif(not FI_available)
def test_imread_uint16():
    expected = np.load(os.path.join(si.data_dir, 'chessboard_GRAY_U8.npy'))
    img = sio.imread(os.path.join(si.data_dir, 'chessboard_GRAY_U16.tif'))
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)


@skipif(not FI_available)
def test_imread_uint16_big_endian():
    expected = np.load(os.path.join(si.data_dir, 'chessboard_GRAY_U8.npy'))
    img = sio.imread(os.path.join(si.data_dir, 'chessboard_GRAY_U16B.tif'))
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)


@skipif(not FI_available)
def test_write_multipage():
    shape = (64, 64, 64)
    x = np.ones(shape, dtype=np.uint8) * np.random.rand(*shape) * 255
    x = x.astype(np.uint8)
    f = NamedTemporaryFile(suffix='.tif')
    fname = f.name
    f.close()
    fi.write_multipage(x, fname)
    y = fi.read_multipage(fname)
    assert_array_equal(x, y)


class TestSave:
    def roundtrip(self, dtype, x, suffix):
        f = NamedTemporaryFile(suffix='.' + suffix)
        fname = f.name
        f.close()
        sio.imsave(fname, x)
        y = sio.imread(fname)
        assert_array_equal(y, x)

    @skipif(not FI_available)
    def test_imsave_roundtrip(self):
        for shape, dtype, format in [
              [(10, 10), (np.uint8, np.uint16), ('tif', 'png')],
              [(10, 10), (np.float32,), ('tif',)],
              [(10, 10, 3), (np.uint8, np.uint16), ('png',)],
              [(10, 10, 4), (np.uint8, np.uint16), ('png',)]
            ]:
            tests = [(d, f) for d in dtype for f in format]
            for d, f in tests:
                x = np.ones(shape, dtype=d) * np.random.rand(*shape)
                if not np.issubdtype(d, float):
                    x = (x * 255).astype(d)
                yield self.roundtrip, d, x, f


@skipif(not FI_available)
def test_metadata():
    meta = fi.read_metadata(os.path.join(si.data_dir, 'multipage.tif'))
    assert meta[('EXIF_MAIN', 'Orientation')] == 1
    assert meta[('EXIF_MAIN', 'Software')].startswith('I')

    meta = fi.read_multipage_metadata(os.path.join(si.data_dir,
                                                   'multipage.tif'))
    assert len(meta) == 2
    assert meta[0][('EXIF_MAIN', 'Orientation')] == 1
    assert meta[1][('EXIF_MAIN', 'Software')].startswith('I')


@skipif(not FI_available)
def test_collection():
    pattern = [os.path.join(data_dir, pic)
               for pic in ['camera.png', 'color.png', 'multipage.tif']]
    images = sio.ImageCollection(pattern[:-1])
    assert len(images) == 2
    assert len(images[:]) == 2

    images = sio.ImageCollection(pattern)
    assert len(images) == 3
    assert len(images[:]) == 3

if __name__ == "__main__":
    run_module_suite()
