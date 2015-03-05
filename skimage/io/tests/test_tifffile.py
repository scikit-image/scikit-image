import os
import skimage as si
import skimage.io as sio
import numpy as np

from numpy.testing import (
    assert_array_equal, assert_array_almost_equal, run_module_suite)

from tempfile import NamedTemporaryFile

_plugins = sio.plugin_order()
sio.use_plugin('tifffile')


np.random.seed(0)


def teardown():
    sio.reset_plugins()


def test_imread_uint16():
    expected = np.load(os.path.join(si.data_dir, 'chessboard_GRAY_U8.npy'))
    img = sio.imread(os.path.join(si.data_dir, 'chessboard_GRAY_U16.tif'))
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)


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

    def test_imsave_roundtrip(self):
        for shape in [(10, 10), (10, 10, 3), (10, 10, 4)]:
            for dtype in (np.uint8, np.uint16, np.float32, np.int16,
                          np.float64):
                x = np.random.rand(*shape)

                if not np.issubdtype(dtype, float):
                    x = (x * np.iinfo(dtype).max).astype(dtype)
                else:
                    x = x.astype(dtype)
                yield self.roundtrip, dtype, x


if __name__ == "__main__":
    run_module_suite()
