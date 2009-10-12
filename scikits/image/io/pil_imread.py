__all__ = ['imread']

import numpy as np

def imread(fname, flatten=False, dtype=None):
    """Load an image from file.

    Parameters
    ----------
    fname : string
        Image file name, e.g. ``test.jpg``.
    flatten : bool
        If true, convert the output to grey-scale.
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
    if flatten:
        im = im.convert('F')
    return np.array(im, dtype=dtype)
