import pathlib
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from _skimage2._shared.testing import fetch
from skimage.io import imread, imsave


def test_imread_uint16():
    expected = np.load(fetch('data/chessboard_GRAY_U8.npy'))
    img = imread(fetch('data/chessboard_GRAY_U16.tif'))
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)


def test_imread_uint16_big_endian():
    expected = np.load(fetch('data/chessboard_GRAY_U8.npy'))
    img = imread(fetch('data/chessboard_GRAY_U16B.tif'))
    assert img.dtype.type == np.uint16
    assert_array_almost_equal(img, expected)


def test_imread_multipage_rgb_tif():
    img = imread(fetch('data/multipage_rgb.tif'))
    assert img.shape == (2, 10, 10, 3), img.shape


def test_imread_handle():
    expected = np.load(fetch('data/chessboard_GRAY_U8.npy'))
    with open(fetch('data/chessboard_GRAY_U16.tif'), 'rb') as fh:
        img = imread(fh)
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)


class TestSave:
    def roundtrip(self, dtype, x, use_pathlib=False, **kwargs):
        with NamedTemporaryFile(suffix='.tif') as f:
            fname = f.name

        if use_pathlib:
            fname = pathlib.Path(fname)
        imsave(fname, x, check_contrast=False, **kwargs)
        y = imread(fname)
        assert_array_equal(x, y)

    shapes_seeds = (
        ((10, 10), 2500279270),
        ((10, 10, 3), 2439842967),
        ((10, 10, 4), 337224809),
    )
    dtypes = (np.uint8, np.uint16, np.float32, np.int16, np.float64)

    @pytest.mark.parametrize("shape,seed", shapes_seeds)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("use_pathlib", [False, True])
    def test_imsave_roundtrip(self, shape, seed, dtype, use_pathlib):
        rng = np.random.RandomState(seed)
        x = rng.rand(*shape)

        if not np.issubdtype(dtype, np.floating):
            x = (x * np.iinfo(dtype).max).astype(dtype)
        else:
            x = x.astype(dtype)
        self.roundtrip(dtype, x, use_pathlib)
