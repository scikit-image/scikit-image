__all__ = ['imread']

import numpy as np

try:
    from PIL import Image
except ImportError:
    raise ImportError("The Python Image Library could not be found. "
                      "Please refer to http://pypi.python.org/pypi/PIL/ "
                      "for further instructions.")

from skimage.util import img_as_ubyte


def imread(fname, dtype=None):
    """Load an image from file.

    """
    im = Image.open(fname)
    if im.mode == 'P':
        if _palette_is_grayscale(im):
            im = im.convert('L')
        else:
            im = im.convert('RGB')
    elif im.mode == '1':
        im = im.convert('L')
    elif im.mode.startswith('I;16'):
        shape = im.size
        dtype = '>u2' if im.mode.endswith('B') else '<u2'
        im = np.fromstring(im.tostring(), dtype)
        im.shape = shape[::-1]
    elif 'A' in im.mode:
        im = im.convert('RGBA')

    return np.array(im, dtype=dtype)


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
    assert pil_image.mode == 'P'
    # get palette as an array with R, G, B columns
    palette = np.asarray(pil_image.getpalette()).reshape((256, 3))
    # Not all palette colors are used; unused colors have junk values.
    start, stop = pil_image.getextrema()
    valid_palette = palette[start:stop]
    # Image is grayscale if channel differences (R - G and G - B)
    # are all zero.
    return np.allclose(np.diff(valid_palette), 0)


def imsave(fname, arr, format_str=None):
    """Save an image to disk.

    Parameters
    ----------
    fname : str or file-like object
        Name of destination file.
    arr : ndarray of uint8 or float
        Array (image) to save.  Arrays of data-type uint8 should have
        values in [0, 255], whereas floating-point arrays must be
        in [0, 1].
    format_str: str
        Format to save as, this is required if using a file-like object;
        this is optional if fname is a string and the format can be
        derived from the extension.

    Notes
    -----
    Currently, only 8-bit precision is supported.

    """
    arr = np.asarray(arr).squeeze()

    if arr.ndim not in (2, 3):
        raise ValueError("Invalid shape for image array: %s" % arr.shape)

    if arr.ndim == 3:
        if arr.shape[2] not in (3, 4):
            raise ValueError("Invalid number of channels in image array.")

    # Image is floating point, assume in [0, 1]
    if np.issubdtype(arr.dtype, float):
        arr = arr * 255

    arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        mode = 'L'

    elif arr.shape[2] in (3, 4):
        mode = {3: 'RGB', 4: 'RGBA'}[arr.shape[2]]

        # Force all integers to bytes
        arr = arr.astype(np.uint8)

    img = Image.fromstring(mode, (arr.shape[1], arr.shape[0]), arr.tostring())
    img.save(fname, format=format_str)


def imshow(arr):
    """Display an image, using PIL's default display command.

    Parameters
    ----------
    arr : ndarray
       Image to display.  Images of dtype float are assumed to be in
       [0, 1].  Images of dtype uint8 are in [0, 255].

    """
    Image.fromarray(img_as_ubyte(arr)).show()


def _app_show():
    pass
