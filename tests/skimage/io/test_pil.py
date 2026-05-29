import os
from io import BytesIO
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from PIL import Image
from _skimage2._shared import testing
from _skimage2._shared._tempfile import temporary_file
from _skimage2._shared.testing import (
    assert_array_almost_equal,
    assert_equal,
    fetch,
)
from skimage.util import img_as_ubyte, img_as_uint
from skimage.io import imread, imsave


def pil_to_ndarray(image, dtype=None, img_num=None):
    """Import a PIL Image object to an ndarray, in memory.

    Parameters
    ----------
    Refer to ``imread``.

    """
    # PIL 12.1.0 renames getdata
    if hasattr(image, "get_flattened_data"):
        image.getdata = image.get_flattened_data

    try:
        # this will raise an IOError if the file is not readable
        image.getdata()[0]
    except OSError as e:
        site = "http://pillow.readthedocs.org/en/latest/installation.html#external-libraries"
        pillow_error_message = str(e)
        error_message = (
            f"Could not load '{image.filename}' \n"
            f"Reason: '{pillow_error_message}'\n"
            f"Please see documentation at: {site}"
        )
        raise ValueError(error_message)
    frames = []
    grayscale = None
    i = 0
    while 1:
        try:
            image.seek(i)
        except EOFError:
            break

        frame = image

        if img_num is not None and img_num != i:
            image.getdata()[0]
            i += 1
            continue

        if image.format == 'PNG' and image.mode == 'I' and dtype is None:
            dtype = 'uint16'

        if image.mode == 'P':
            if grayscale is None:
                grayscale = _palette_is_grayscale(image)

            if grayscale:
                frame = image.convert('L')
            else:
                if image.format == 'PNG' and 'transparency' in image.info:
                    frame = image.convert('RGBA')
                else:
                    frame = image.convert('RGB')

        elif image.mode == '1':
            frame = image.convert('L')

        elif 'A' in image.mode:
            frame = image.convert('RGBA')

        elif image.mode == 'CMYK':
            frame = image.convert('RGB')

        if image.mode.startswith('I;16'):
            shape = image.size
            dtype = '>u2' if image.mode.endswith('B') else '<u2'
            if 'S' in image.mode:
                dtype = dtype.replace('u', 'i')
            frame = np.frombuffer(frame.tobytes(), dtype)
            frame = np.reshape(frame, shape[::-1], copy=False)

        else:
            frame = np.array(frame, dtype=dtype)

        frames.append(frame)
        i += 1

        if img_num is not None:
            break

    if hasattr(image, 'fp') and image.fp:
        image.fp.close()

    if img_num is None and len(frames) > 1:
        return np.array(frames)
    elif frames:
        return frames[0]
    elif img_num:
        raise IndexError(f'Could not find image  #{img_num}')


def _palette_is_grayscale(pil_image):
    """Return True if PIL image in palette mode is grayscale.

    Parameters
    ----------
    pil_image : PIL image
        PIL Image that is in Palette mode.

    Returns
    -------
    is_grayscale : bool
        True if all colors in image palette are gray.
    """
    if pil_image.mode != 'P':
        raise ValueError('pil_image.mode must be equal to "P".')
    # get palette as an array with R, G, B columns
    # Starting in pillow 9.1 palettes may have less than 256 entries
    palette = np.asarray(pil_image.getpalette()).reshape((-1, 3))
    # Not all palette colors are used; unused colors have junk values.
    start, stop = pil_image.getextrema()
    valid_palette = palette[start : stop + 1]
    # Image is grayscale if channel differences (R - G and G - B)
    # are all zero.
    return np.allclose(np.diff(valid_palette), 0)


def ndarray_to_pil(arr, format_str=None):
    """Export an ndarray to a PIL object.

    Parameters
    ----------
    Refer to ``imsave``.

    """
    if arr.ndim == 3:
        arr = img_as_ubyte(arr)
        mode = {3: 'RGB', 4: 'RGBA'}[arr.shape[2]]

    elif format_str in ['png', 'PNG']:
        mode = 'I;16'

        if arr.dtype.kind == 'f':
            arr = img_as_uint(arr)

        elif arr.max() < 256 and arr.min() >= 0:
            arr = arr.astype(np.uint8)
            mode = 'L'

        else:
            arr = img_as_uint(arr)

    else:
        arr = img_as_ubyte(arr)
        mode = 'L'

    try:
        array_buffer = arr.tobytes()
    except AttributeError:
        array_buffer = arr.tostring()  # Numpy < 1.9

    if arr.ndim == 2:
        im = Image.new(mode, arr.T.shape)
        try:
            im.frombytes(array_buffer, 'raw', mode)
        except AttributeError:
            im.fromstring(array_buffer, 'raw', mode)  # PIL 1.1.7
    else:
        image_shape = (arr.shape[1], arr.shape[0])
        try:
            im = Image.frombytes(mode, image_shape, array_buffer)
        except AttributeError:
            im = Image.fromstring(mode, image_shape, array_buffer)  # PIL 1.1.7
    return im


def test_imread_as_gray():
    img = imread(fetch('data/color.png'), as_gray=True)
    assert img.ndim == 2
    assert img.dtype == np.float64
    img = imread(fetch('data/camera.png'), as_gray=True)
    # check that conversion does not happen for a gray image
    assert np.dtype(img.dtype).char in np.typecodes['AllInteger']


@pytest.mark.parametrize('explicit_kwargs', [False, True])
def test_imread_separate_channels(explicit_kwargs):
    # Test that imread returns RGB(A) values contiguously even when they are
    # stored in separate planes.
    x = np.random.RandomState(819070535).rand(3, 16, 8)
    with NamedTemporaryFile(suffix='.tif') as f:
        fname = f.name

    # Tifffile is used as backend whenever suffix is .tif or .tiff
    # To avoid pending changes to tifffile defaults, we must specify this is an
    # RGB image with separate planes (i.e., channel_axis=0).
    if explicit_kwargs:
        pass
    else:
        pass

    imsave(fname, x)
    img = imread(fname)
    os.remove(fname)
    assert img.shape == (16, 8, 3), img.shape


def test_imread_multipage_rgb_tif():
    img = imread(fetch('data/multipage_rgb.tif'))
    assert img.shape == (2, 10, 10, 3), img.shape


def test_palette_is_gray():
    gray = Image.open(fetch('data/palette_gray.png'))
    assert _palette_is_grayscale(gray)
    color = Image.open(fetch('data/palette_color.png'))
    assert not _palette_is_grayscale(color)


def test_imread_uint16():
    expected = np.load(fetch('data/chessboard_GRAY_U8.npy'))
    img = imread(fetch('data/chessboard_GRAY_U16.tif'))
    assert np.issubdtype(img.dtype, np.uint16)
    assert_array_almost_equal(img, expected)


def test_imread_truncated_jpg():
    with testing.raises(IOError):
        imread(fetch('data/truncated.jpg'))


def test_imread_uint16_big_endian():
    expected = np.load(fetch('data/chessboard_GRAY_U8.npy'))
    img = imread(fetch('data/chessboard_GRAY_U16B.tif'))
    assert img.dtype.type == np.uint16
    assert_array_almost_equal(img, expected)


class TestSave:
    def roundtrip_file(self, x):
        with temporary_file(suffix='.png') as fname:
            imsave(fname, x)
            y = imread(fname)
            return y

    def roundtrip_pil_image(self, x):
        pil_image = ndarray_to_pil(x)
        y = pil_to_ndarray(pil_image)
        return y

    def verify_roundtrip(self, dtype, x, y, scaling=1):
        assert_array_almost_equal((x * scaling).astype(np.int32), y)

    def verify_imsave_roundtrip(self, roundtrip_function):
        rng = np.random.RandomState(2316108381)
        for shape in [(10, 10), (10, 10, 3), (10, 10, 4)]:
            for dtype in (np.uint8, np.uint16, np.float32, np.float64):
                x = np.ones(shape, dtype=dtype) * rng.rand(*shape)

                if np.issubdtype(dtype, np.floating):
                    yield (self.verify_roundtrip, dtype, x, roundtrip_function(x), 255)
                else:
                    x = (x * 255).astype(dtype)
                    yield (self.verify_roundtrip, dtype, x, roundtrip_function(x))

    def test_imsave_roundtrip_file(self):
        self.verify_imsave_roundtrip(self.roundtrip_file)

    def test_imsave_roundtrip_pil_image(self):
        self.verify_imsave_roundtrip(self.roundtrip_pil_image)


def test_imexport_imimport():
    shape = (2, 2)
    image = np.zeros(shape)
    pil_image = ndarray_to_pil(image)
    out = pil_to_ndarray(pil_image)
    assert_equal(out.shape, shape)


def test_extreme_palette():
    img = imread(fetch('data/green_palette.png'))
    assert_equal(img.ndim, 3)
