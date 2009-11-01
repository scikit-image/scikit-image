__all__ = ['imread']

import numpy as np
import plugin

try:
    from PIL import Image
    has_pil = True
except ImportError:
    has_pil = False

def imread(fname, as_grey=False, dtype=None):
    """Load an image from file.

    """
    im = Image.open(fname)
    if im.mode == 'P':
        if palette_is_grayscale(im):
            im = im.convert('L')
        else:
            im = im.convert('RGB')

    if as_grey and not \
           im.mode in ('1', 'L', 'I', 'F', 'I;16', 'I;16L', 'I;16B'):
        im = im.convert('F')

    return np.array(im, dtype=dtype)

def palette_is_grayscale(pil_image):
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


if has_pil:
    plugin.register('PIL', read=imread)
