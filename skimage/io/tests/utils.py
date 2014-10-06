from tempfile import NamedTemporaryFile

from skimage import (
    data, io, img_as_uint, img_as_bool, img_as_float, img_as_int, img_as_ubyte)
from numpy import testing
import numpy as np


def roundtrip(img, plugin, suffix):
    """Save and read an image using a specified plugin"""
    if not '.' in suffix:
        suffix = '.' + suffix
    temp_file = NamedTemporaryFile(suffix=suffix, delete=False)
    temp_file.close()
    fname = temp_file.name
    io.imsave(fname, img, plugin=plugin)
    return io.imread(fname, plugin=plugin)


def ubyte_check(plugin, fmt='png'):
    """Check roundtrip behavior for images that can only be saved as uint8

    All major input types should be handled as ubytes and read
    back correctly.
    """
    img = img_as_ubyte(data.chelsea())
    r1 = roundtrip(img, plugin, fmt)
    testing.assert_allclose(img, r1)

    img2 = img > 128
    r2 = roundtrip(img2, plugin, fmt)
    testing.assert_allclose(img2.astype(np.uint8), r2)

    img3 = img_as_float(img)
    r3 = roundtrip(img3, plugin, fmt)
    testing.assert_allclose(r3, img)

    img4 = img_as_int(img)
    r4 = roundtrip(img4, plugin, fmt)
    testing.assert_allclose(r4, img)

    img5 = img_as_uint(img)
    r5 = roundtrip(img5, plugin, fmt)
    testing.assert_allclose(r5, img)


def full_range_check(plugin, fmt='png'):
    """Check the roundtrip behavior for images that support most types.

    All major input types should be handled, except bool is treated
    as ubyte and float can treated as uint16 or float.
    """

    img = img_as_ubyte(data.moon())
    r1 = roundtrip(img, plugin, fmt)
    testing.assert_allclose(img, r1)

    img2 = img > 128
    r2 = roundtrip(img2, plugin, fmt)
    testing.assert_allclose(img2.astype(np.uint8), r2)

    img3 = img_as_float(img)
    r3 = roundtrip(img3, plugin, fmt)
    if r3.dtype.kind == 'f':
        testing.assert_allclose(img3, r3)
    else:
        testing.assert_allclose(r3, img_as_uint(img))

    img4 = img_as_int(img)
    r4 = roundtrip(img4, plugin, fmt)
    testing.assert_allclose(r4, img4)

    img5 = img_as_uint(img)
    r5 = roundtrip(img5, plugin, fmt)
    testing.assert_allclose(r5, img5)


if __name__ == '__main__':
    ubyte_check('pil')
    full_range_check('pil')
    ubyte_check('pil', 'bmp')
    full_range_check('pil', 'tiff')
