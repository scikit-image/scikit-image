import os
import skimage as si
import skimage.io as sio
import numpy as np

from numpy.testing import *
from numpy.testing.decorators import skipif
from tempfile import NamedTemporaryFile

try:
    import skimage.io._plugins.tifffile_plugin as tf
    _plugins = sio.plugin_order()
    TF_available = True
    sio.use_plugin('tifffile')
except ImportError:
    TF_available = False


def teardown():
    sio.reset_plugins()


@skipif(not TF_available)
def test_imread_uint16():
    expected = np.load(os.path.join(si.data_dir, 'chessboard_GRAY_U8.npy'))
    img = sio.imread(os.path.join(si.data_dir, 'chessboard_GRAY_U16.tif'))
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)


@skipif(not TF_available)
def test_imread_uint16_big_endian():
    expected = np.load(os.path.join(si.data_dir, 'chessboard_GRAY_U8.npy'))
    img = sio.imread(os.path.join(si.data_dir, 'chessboard_GRAY_U16B.tif'))
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)


class TestSave:
    def roundtrip(self, dtype, x):
        f = NamedTemporaryFile(suffix='.tif')
        fname = f.name
        f.close()
        sio.imsave(fname, x)
        y = sio.imread(fname)
        assert_array_equal(x, y)

    @skipif(not TF_available)
    def test_imsave_roundtrip(self):
        for shape in [(10, 10), (10, 10, 3), (10, 10, 4)]:
            for dtype in (np.uint8, np.uint16, np.float32, np.float64):
                x = np.ones(shape, dtype=dtype) * np.random.random(shape)

                if not np.issubdtype(dtype, float):
                    x = (x * 255).astype(dtype)
                yield self.roundtrip, dtype, x


if __name__ == "__main__":
    run_module_suite()
