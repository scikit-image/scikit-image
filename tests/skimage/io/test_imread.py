from tempfile import NamedTemporaryFile

import numpy as np
from skimage import io
from skimage.io import imread, imsave

from _skimage2._shared import testing
from _skimage2._shared.testing import (
    TestCase,
    assert_array_equal,
    assert_array_almost_equal,
    fetch,
)


def test_imread_as_gray():
    img = imread(fetch('data/color.png'), as_gray=True)
    assert img.ndim == 2
    assert img.dtype == np.float64
    img = imread(fetch('data/camera.png'), as_gray=True)
    # check that conversion does not happen for a gray image
    assert np.dtype(img.dtype).char in np.typecodes['AllInteger']


def test_imread_palette():
    img = imread(fetch('data/palette_color.png'))
    assert img.ndim == 3


def test_imread_truncated_jpg():
    with testing.raises(OSError, match="Truncated File Read"):
        io.imread(fetch('data/truncated.jpg'))


def test_bilevel():
    expected = np.zeros((10, 10), bool)
    expected[::2] = 1

    img = imread(fetch('data/checker_bilevel.png'))
    assert_array_equal(img.astype(bool), expected)


class TestSave(TestCase):
    def roundtrip(self, x, scaling=1):
        with NamedTemporaryFile(suffix='.png') as f:
            fname = f.name

        imsave(fname, x)
        y = imread(fname)

        assert_array_almost_equal((x * scaling).astype(np.int32), y)

    def test_imsave_roundtrip(self):
        dtype = np.uint8
        rng = np.random.RandomState(3174584926)
        for shape in [(10, 10), (10, 10, 3), (10, 10, 4)]:
            x = np.ones(shape, dtype=dtype) * rng.rand(*shape)

            if np.issubdtype(dtype, np.floating):
                self.roundtrip(x, 255)
            else:
                x = (x * 255).astype(dtype)
                self.roundtrip(x)
