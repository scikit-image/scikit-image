import os
import pathlib
import tempfile

import numpy as np
from numpy.testing import assert_allclose
import pytest

from skimage import io
from _skimage2._shared.testing import assert_array_equal, fetch, assert_stacklevel
from _skimage2._shared._dependency_checks import is_wasm
from skimage.data import data_dir


one_by_one_jpeg = (
    b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01'
    b'\x00\x01\x00\x00\xff\xdb\x00C\x00\x03\x02\x02\x02\x02'
    b'\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04'
    b'\x04\x04\x04\x08\x06\x06\x05\x06\t\x08\n\n\t\x08\t\t'
    b'\n\x0c\x0f\x0c\n\x0b\x0e\x0b\t\t\r\x11\r\x0e\x0f\x10'
    b'\x10\x11\x10\n\x0c\x12\x13\x12\x10\x13\x0f\x10\x10'
    b'\x10\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11'
    b'\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\t\xff\xc4\x00'
    b'\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00'
    b'\x00?\x00*\x9f\xff\xd9'
)


def test_stack_basic():
    x = np.arange(12).reshape(3, 4)
    io.push(x)

    assert_array_equal(io.pop(), x)


def test_stack_non_array():
    with pytest.raises(ValueError):
        io.push([[1, 2, 3]])


# skimage.io.imread ------------------------------------------------------------


def test_imread_file_url():
    # tweak data path so that file URI works on both unix and windows.
    data_path = str(fetch('data/camera.png'))
    data_path = data_path.replace(os.path.sep, '/')
    image_url = f'file:///{data_path}'
    image = io.imread(image_url)
    assert image.shape == (512, 512)


@pytest.mark.skipif(is_wasm, reason="no access to pytest-localserver")
def test_imread_http_url(httpserver):
    # httpserver is a fixture provided by pytest-localserver
    # https://bitbucket.org/pytest-dev/pytest-localserver/
    httpserver.serve_content(one_by_one_jpeg)
    # it will serve anything you provide to it on its url.
    # we add a /test.jpg so that we can identify the content
    # by extension
    image = io.imread(httpserver.url + '/test.jpg' + '?' + 's' * 266)
    assert image.shape == (1, 1)


def test_imread_pathlib_tiff():
    """Tests reading from Path object (issue gh-5545)."""

    # read via fetch
    fname = fetch('data/multipage.tif')
    expected = io.imread(fname)

    # read by passing in a pathlib.Path object
    path = pathlib.Path(fname)
    img = io.imread(path)

    assert img.shape == (2, 15, 10)
    assert_array_equal(expected, img)


def _named_tempfile_func(error_class):
    """Create a mock function for NamedTemporaryFile that always raises.

    Parameters
    ----------
    error_class : exception class
        The error that should be raised when asking for a NamedTemporaryFile.

    Returns
    -------
    named_temp_file : callable
        A function that always raises the desired error.

    Notes
    -----
    Although this function has general utility for raising errors, it is
    expected to be used to raise errors that ``tempfile.NamedTemporaryFile``
    from the Python standard library could raise. As of this writing, these
    are ``FileNotFoundError``, ``FileExistsError``, ``PermissionError``, and
    ``BaseException``. See
    `this comment <https://github.com/scikit-image/scikit-image/issues/3785#issuecomment-486598307>`__
    for more information.
    """

    def named_temp_file(*args, **kwargs):
        raise error_class()

    return named_temp_file


@pytest.mark.parametrize(
    'error_class', [FileNotFoundError, FileExistsError, PermissionError, BaseException]
)
def test_failed_temporary_file(monkeypatch, error_class):
    fetch('data/camera.png')
    # tweak data path so that file URI works on both unix and windows.
    data_path = data_dir.lstrip(os.path.sep)
    data_path = data_path.replace(os.path.sep, '/')
    image_url = f'file:///{data_path}/camera.png'
    with monkeypatch.context():
        monkeypatch.setattr(
            tempfile, 'NamedTemporaryFile', _named_tempfile_func(error_class)
        )
        with pytest.raises(error_class):
            io.imread(image_url)


def test_imread_as_gray():
    img = io.imread(fetch('data/color.png'), as_gray=True)
    assert img.ndim == 2
    assert img.dtype == np.float64
    assert type(img) is np.ndarray

    img = io.imread(fetch('data/camera.png'), as_gray=True)
    # check that conversion does not happen for a gray image
    assert np.dtype(img.dtype).char in np.typecodes['AllInteger']


def test_imread_palette():
    img = io.imread(fetch('data/palette_color.png'))
    assert img.ndim == 3


def test_imread_truncated_jpg():
    # imageio>2.0 uses Pillow / PIL to try and load the file.
    # Oddly, PIL explicitly raises a OSError when the file read fails.
    with pytest.raises(OSError, match=r"Truncated File Read"):
        io.imread(fetch('data/truncated.jpg'))


def test_imread_bilevel():
    expected = np.zeros((10, 10), bool)
    expected[::2] = 1
    img = io.imread(fetch('data/checker_bilevel.png'))
    assert_array_equal(img.astype(bool), expected)


# skimage.io.imsave ------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,dtype",
    [
        # float32, float64 can't be saved as PNG and raise
        # uint32 is not roundtripping properly
        ((10, 10), np.uint8),
        ((10, 10), np.uint16),
        ((10, 10, 2), np.uint8),
        ((10, 10, 3), np.uint8),
        ((10, 10, 4), np.uint8),
    ],
)
def test_imsave_roundtrip(shape, dtype, tmp_path):
    if np.issubdtype(dtype, np.floating):
        min_ = 0
        max_ = 1
    else:
        min_ = 0
        max_ = np.iinfo(dtype).max
    expected = np.linspace(
        min_, max_, endpoint=True, num=np.prod(shape), dtype=dtype
    )
    expected = expected.reshape(shape)
    file_path = tmp_path / "roundtrip.png"
    io.imsave(file_path, expected)
    actual = io.imread(file_path)
    np.testing.assert_array_almost_equal(actual, expected)


@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 3), (10, 10, 4)])
def test_imsave_roundtrip_uint8(shape):
    rng = np.random.RandomState(3174584926)
    img = np.ones(shape, dtype=np.uint8) * rng.rand(*shape)
    img = (img * 255).astype(np.uint8)
    expected = img.astype(np.int32)
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        fname = f.name

    io.imsave(fname, img)
    actual = io.imread(fname)
    assert_allclose(expected, actual)


def test_imsave_bool_array():
    a = np.zeros((5, 5), bool)
    a[2, 2] = True
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        fname = f.name
    with pytest.warns(UserWarning, match=r'.* is a boolean image') as record:
        io.imsave(fname, a)
    assert_stacklevel(record)
