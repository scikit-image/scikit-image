__all__ = ['imread']

import numpy as np

def imread(fname, flatten=False, dtype=None):
    """Load an image from file.

    Parameters
    ----------
    fname : string
        Image file name, e.g. ``test.jpg``.
    flatten : bool
        If True, convert color images to grey-scale. If `dtype` is not given,
        converted color images are returned as 32-bit float images.
        Images that are already in grey-scale format are not converted.
    dtype : dtype, optional
        NumPy data-type specifier. If given, the returned image has this type.
        If None (default), the data-type is determined automatically.

    Returns
    -------
    img_array : ndarray
        The different colour bands/channels are stored in the
        third dimension, such that a grey-image is MxN, an
        RGB-image MxNx3 and an RGBA-image MxNx4.

    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Could not import the Python Imaging Library (PIL)"
                          " required to load image files.  Please refer to"
                          " http://pypi.python.org/pypi/PIL/ for installation"
                          " instructions.")

    im = Image.open(fname)
    if flatten and not im.mode in ('1', 'L', 'I', 'F', 'I;16', 'I;16L', 'I;16B'):
        im = im.convert('F')
    return np.array(im, dtype=dtype)
