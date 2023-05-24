import numpy as np
import pytest

from skimage.io import imread, imsave, use_plugin, reset_plugins
from skimage._shared import testing


pytest.importorskip('SimpleITK')

np.random.seed(0)


def teardown():
    reset_plugins()


@pytest.fixture(autouse=True)
def setup_plugin():
    """This ensures that `use_plugin` is directly called before all tests to
    ensure that SimpleITK is used.
    """
    use_plugin('simpleitk')
    yield


def test_imread_as_gray():
    img = imread(testing.fetch('data/color.png'), as_gray=True)
    assert img.ndim == 2
    assert img.dtype == np.float64
    img = imread(testing.fetch('data/camera.png'), as_gray=True)
    # check that conversion does not happen for a gray image
    assert np.sctype2char(img.dtype) in np.typecodes['AllInteger']


def test_bilevel():
    expected = np.zeros((10, 10))
    expected[::2] = 255

    img = imread(testing.fetch('data/checker_bilevel.png'))
    np.testing.assert_array_equal(img, expected)


def test_imread_truncated_jpg():
    with pytest.raises(RuntimeError):
        imread(testing.fetch('data/truncated.jpg'))


def test_imread_uint16():
    expected = np.load(testing.fetch('data/chessboard_GRAY_U8.npy'))
    img = imread(testing.fetch('data/chessboard_GRAY_U16.tif'))
    assert np.issubdtype(img.dtype, np.uint16)
    np.testing.assert_array_almost_equal(img, expected)


def test_imread_uint16_big_endian():
    expected = np.load(testing.fetch('data/chessboard_GRAY_U8.npy'))
    img = imread(testing.fetch('data/chessboard_GRAY_U16B.tif'))
    np.testing.assert_array_almost_equal(img, expected)


@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 3), (10, 10, 4)])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_imsave_roundtrip(shape, dtype, tmp_path):
    expected = np.zeros(shape, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        info_func = np.finfo
    else:
        info_func = np.iinfo
    expected.flat[0] = info_func(dtype).min
    expected.flat[-1] = info_func(dtype).max
    file_path = tmp_path / "roundtrip.mha"
    imsave(file_path, expected)
    actual = imread(file_path)
    np.testing.assert_array_almost_equal(actual, expected)
