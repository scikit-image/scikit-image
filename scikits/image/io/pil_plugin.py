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
    if as_grey and \
           not im.mode in ('1', 'L', 'I', 'F', 'I;16', 'I;16L', 'I;16B'):
        im = im.convert('F')
    return np.array(im, dtype=dtype)

if has_pil:
    plugin.register('PIL', read=imread)
