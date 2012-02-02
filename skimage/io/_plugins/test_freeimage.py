import os
import skimage as si
import skimage.io as sio
import skimage.io._plugins.freeimage_plugin as fi

from numpy.testing import *


def test_read():
    sio.use_plugin('freeimage', 'imread')
    img = sio.imread(os.path.join(si.data_dir, 'color.png'))
    assert img.shape == (370, 371, 3)
    assert all(img[274,135] == [0, 130, 253])

def test_metadata():
    meta = fi.read_metadata(os.path.join(si.data_dir, 'multipage.tif'))
    assert meta[('EXIF_MAIN', 'BitsPerSample')] == 8
    assert meta[('EXIF_MAIN', 'Software')].startswith('ImageMagick')

    meta = fi.read_multipage_metadata(os.path.join(si.data_dir, 'multipage.tif'))
    assert len(meta) == 2
    assert meta[0][('EXIF_MAIN', 'BitsPerSample')] == 8
    assert meta[1][('EXIF_MAIN', 'Software')].startswith('ImageMagick')

    
if __name__ == "__main__":
    run_module_suite()
