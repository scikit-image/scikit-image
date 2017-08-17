import os
import numpy as np

try:
    import skimage as si
except Exception:
    si = None

from numpy.testing import (
    assert_array_equal, assert_array_almost_equal, run_module_suite)
from numpy.testing.decorators import skipif

from tempfile import NamedTemporaryFile
from .tifffile import imread, imsave


np.random.seed(0)


@skipif(si is None)
def test_imread_uint16():
    expected = np.load(os.path.join(si.data_dir, 'chessboard_GRAY_U8.npy'))
    img = imread(os.path.join(si.data_dir, 'chessboard_GRAY_U16.tif'))
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)


@skipif(si is None)
def test_imread_uint16_big_endian():
    expected = np.load(os.path.join(si.data_dir, 'chessboard_GRAY_U8.npy'))
    img = imread(os.path.join(si.data_dir, 'chessboard_GRAY_U16B.tif'))
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)


def test_extension():
    from .tifffile.tifffile import decode_packbits
    import types
    assert isinstance(decode_packbits, types.BuiltinFunctionType), type(decode_packbits)


class TestSave:

    def roundtrip(self, dtype, x):

        # input: file name
        f = NamedTemporaryFile(suffix='.tif')
        fname = f.name
        f.close()
        imsave(fname, x)
        y = imread(fname)
        assert_array_equal(x, y)

        # input: open file object
        f = NamedTemporaryFile(suffix='.tif')
        imsave(f, x)
        f.seek(0)
        y = imread(f)
        assert_array_equal(x, y)
        f.close()

        #input: byte stream
        from io import BytesIO
        b = BytesIO()
        imsave(b, x)
        b.seek(0)
        y = imread(b)
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
