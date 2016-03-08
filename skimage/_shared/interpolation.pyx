from interpolation cimport coord_map, get_pixel2d
import numpy as np
cimport numpy as cnp
from .utils import _mode_deprecations


def coord_map_py(Py_ssize_t dim, long coord, mode):
    """Python wrapper for `interpolation.coord_map`."""
    cdef char mode_c = ord(mode[0].upper())
    return coord_map(dim, coord, mode_c)


def extend_image(image, pad=10, mode='constant', cval=0):
    """Pad a 2D image by `pad` pixels on each side.

    Parameters
    ----------
    image : ndarray
        Input image.
    pad : int, optional
        The number of pixels to pad around the border
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    extended : ndarray
        The extended version of the input image.

    Notes
    -----
    For image padding, `skimage.util.pad` should be used instead.  This
    function is intended only for testing `get_pixel2d` and demonstrating the
    coordinate mapping modes implemented in `coord_map`.
    """
    mode = _mode_deprecations(mode)
    cdef:
        Py_ssize_t rows = image.shape[0]
        Py_ssize_t cols = image.shape[1]
        long ro, co
        char mode_c = ord(mode[0].upper())

    image = np.ascontiguousarray(image.astype(np.float64))
    output_shape = np.asarray(image.shape) + 2 * pad
    extended = np.zeros(output_shape, dtype=image.dtype)
    for r in range(-pad, rows + pad):
        for c in range(-pad, cols + pad):
            ro = r + pad
            co = c + pad
            extended[ro, co] = get_pixel2d(<double*> cnp.PyArray_DATA(image),
                                           rows, cols, <long> r, <long> c,
                                           mode_c, <double> cval)
    return extended
