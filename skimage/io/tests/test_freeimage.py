import os
import skimage as si
import skimage.io as sio

from numpy.testing import *
from numpy.testing.decorators import skipif

try:
    import skimage.io._plugins.freeimage_plugin as fi
    FI_available = True
    sio.use_plugin('freeimage')
except OSError:
    FI_available = False

@skipif(not FI_available)
def test_imread():
    img = sio.imread(os.path.join(si.data_dir, 'color.png'))
    assert img.shape == (370, 371, 3)
    assert all(img[274,135] == [0, 130, 253])

@skipif(not FI_available)
def test_imread_uint16():
    expected = np.load(os.path.join(data_dir, 'chessboard_GRAY_U8.npy'))
    img = sio.imread(os.path.join(data_dir, 'chessboard_GRAY_U16.tif'))
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)

@skipif(not FI_available)
def test_imread_uint16_big_endian():
    expected = np.load(os.path.join(data_dir, 'chessboard_GRAY_U8.npy'))
    img = sio.imread(os.path.join(data_dir, 'chessboard_GRAY_U16B.tif'))
    assert img.dtype == np.dtype('>u2')
    assert_array_almost_equal(img, expected)


class TestSave:
    def roundtrip(self, dtype, x, scaling=1):
        f = NamedTemporaryFile(suffix='.tif')
        fname = f.name
        f.close()
        sio.imsave(fname, x)
        y = sio.imread(fname)

        assert_array_almost_equal((x * scaling).astype(np.int32), y)

    @skipif(not FI_available)
    def test_imsave_roundtrip(self):
        for shape in [(10, 10), (10, 10, 3), (10, 10, 4)]:
            for dtype in (np.uint8, np.uint16, np.float32, np.float64):
                x = np.ones(shape, dtype=dtype) * np.random.random(shape)

                if np.issubdtype(dtype, float):
                    yield self.roundtrip, dtype, x, 255
                else:
                    x = (x * 255).astype(dtype)
                    yield self.roundtrip, dtype, x


@skipif(not FI_available)
def test_metadata():
    meta = fi.read_metadata(os.path.join(si.data_dir, 'multipage.tif'))
    assert meta[('EXIF_MAIN', 'Orientation')] == 1
    assert meta[('EXIF_MAIN', 'Software')].startswith('ImageMagick')

    meta = fi.read_multipage_metadata(os.path.join(si.data_dir, 'multipage.tif'))
    assert len(meta) == 2
    assert meta[0][('EXIF_MAIN', 'Orientation')] == 1
    assert meta[1][('EXIF_MAIN', 'Software')].startswith('ImageMagick')


if __name__ == "__main__":
    run_module_suite()
