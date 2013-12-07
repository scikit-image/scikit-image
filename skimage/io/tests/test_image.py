from io import BytesIO

import numpy as np
from skimage import img_as_ubyte
from skimage.io import Image, imread

from numpy.testing import assert_equal, assert_array_equal


def test_tags():
    f = Image([1, 2, 3], foo='bar', sigma='delta')
    g = Image([3, 2, 1], sun='moon')
    h = Image([1, 1, 1])

    assert_equal(f.tags['foo'], 'bar')
    assert_array_equal((g + 2).tags['sun'], 'moon')
    assert_equal(h.tags, {})


def test_repr_png_roundtrip():
    # Use RGB-like shape since some backends convert grayscale to RGB
    original_array = 255 * np.ones((5, 5, 3), dtype=np.uint8)
    image = Image(original_array)
    array = imread(BytesIO(image._repr_png_()))
    # Force output to ubyte range for plugin compatibility.
    # For example, Matplotlib will return floats even if the image is uint8.
    assert_array_equal(img_as_ubyte(array), original_array)
    # Note that PIL breaks with `_repr_jpeg_`.


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
