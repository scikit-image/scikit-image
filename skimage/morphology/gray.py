"""
Grayscale morphological operations
"""
import functools

import numpy as np
from scipy import ndimage as ndi

from ..util import crop
from .footprints import _footprint_is_sequence, _shape_from_sequence
from .misc import default_footprint

__all__ = ['erosion', 'dilation', 'opening', 'closing', 'white_tophat',
           'black_tophat']


def _iterate_gray_func(gray_func, image, footprints, out):
    """Helper to call `binary_func` for each footprint in a sequence.

    binary_func is a binary morphology function that accepts "structure",
    "output" and "iterations" keyword arguments
    (e.g. `scipy.ndimage.binary_erosion`).
    """
    fp, num_iter = footprints[0]
    gray_func(image, footprint=fp, output=out)
    for _ in range(1, num_iter):
        gray_func(out.copy(), footprint=fp, output=out)
    for fp, num_iter in footprints[1:]:
        # Note: out.copy() because the computation cannot be in-place!
        for _ in range(num_iter):
            gray_func(out.copy(), footprint=fp, output=out)
    return out


def _shift_footprint(footprint, shift_x, shift_y):
    """Shift the binary image `footprint` in the left and/or up.

    This only affects 2D footprints with even number of rows
    or columns.

    Parameters
    ----------
    footprint : 2D array, shape (M, N)
        The input footprint.
    shift_x, shift_y : bool
        Whether to move `footprint` along each axis.

    Returns
    -------
    out : 2D array, shape (M + int(shift_x), N + int(shift_y))
        The shifted footprint.
    """
    if footprint.ndim != 2:
        # do nothing for 1D or 3D or higher footprints
        return footprint
    m, n = footprint.shape
    if m % 2 == 0:
        extra_row = np.zeros((1, n), footprint.dtype)
        if shift_x:
            footprint = np.vstack((footprint, extra_row))
        else:
            footprint = np.vstack((extra_row, footprint))
        m += 1
    if n % 2 == 0:
        extra_col = np.zeros((m, 1), footprint.dtype)
        if shift_y:
            footprint = np.hstack((footprint, extra_col))
        else:
            footprint = np.hstack((extra_col, footprint))
    return footprint


def _invert_footprint(footprint):
    """Change the order of the values in `footprint`.

    This is a patch for the *weird* footprint inversion in
    `ndi.grey_morphology` [1]_.

    Parameters
    ----------
    footprint : array
        The input footprint.

    Returns
    -------
    inverted : array, same shape and type as `footprint`
        The footprint, in opposite order.

    Examples
    --------
    >>> footprint = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], np.uint8)
    >>> _invert_footprint(footprint)
    array([[1, 1, 0],
           [1, 1, 0],
           [0, 0, 0]], dtype=uint8)

    References
    ----------
    .. [1] https://github.com/scipy/scipy/blob/ec20ababa400e39ac3ffc9148c01ef86d5349332/scipy/ndimage/morphology.py#L1285  # noqa
    """
    inverted = footprint[(slice(None, None, -1),) * footprint.ndim]
    return inverted


def pad_for_eccentric_footprints(func):
    """Pad input images for certain morphological operations.

    Parameters
    ----------
    func : callable
        A morphological function, either opening or closing, that
        supports eccentric footprints. Its parameters must
        include at least `image`, `footprint`, and `out`.

    Returns
    -------
    func_out : callable
        The same function, but correctly padding the input image before
        applying the input function.

    See Also
    --------
    opening, closing.
    """
    @functools.wraps(func)
    def func_out(image, footprint, out=None, *args, **kwargs):
        pad_widths = []
        padding = False
        if out is None:
            out = np.empty_like(image)
        if _footprint_is_sequence(footprint):
            # Note: in practice none of our built-in footprint sequences will
            #       require padding (all are symmetric and have odd sizes)
            footprint_shape = _shape_from_sequence(footprint)
        else:
            footprint_shape = footprint.shape
        for axis_len in footprint_shape:
            if axis_len % 2 == 0:
                axis_pad_width = axis_len - 1
                padding = True
            else:
                axis_pad_width = 0
            pad_widths.append((axis_pad_width,) * 2)
        if padding:
            image = np.pad(image, pad_widths, mode='edge')
            out_temp = np.empty_like(image)
        else:
            out_temp = out
        out_temp = func(image, footprint, out=out_temp, *args, **kwargs)
        if padding:
            out[:] = crop(out_temp, pad_widths)
        else:
            out = out_temp
        return out
    return func_out


@default_footprint
def erosion(image, footprint=None, out=None, shift_x=False, shift_y=False):
    """Return grayscale morphological erosion of an image.

    Morphological erosion sets a pixel at (i,j) to the minimum over all pixels
    in the neighborhood centered at (i,j). Erosion shrinks bright regions and
    enlarges dark regions.

    Parameters
    ----------
    image : ndarray
        Image array.
    footprint : ndarray or tuple, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped footprint (connectivity=1). The footprint
        can also be provided as a sequence of smaller footprints as described
        in the notes below.
    out : ndarrays, optional
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    shift_x, shift_y : bool, optional
        shift footprint about center point. This only affects
        eccentric footprints (i.e. footprint with even numbered
        sides).

    Returns
    -------
    eroded : array, same shape as `image`
        The result of the morphological erosion.

    Notes
    -----
    For ``uint8`` (and ``uint16`` up to a certain bit-depth) data, the
    lower algorithm complexity makes the `skimage.filters.rank.minimum`
    function more efficient for larger images and footprints.

    The footprint can also be a provided as a sequence of 2-tuples where the
    first element of each 2-tuple is a footprint ndarray and the second element
    is an integer describing the number of times it should be iterated. For
    example ``footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]``
    would apply a 9x1 footprint followed by a 1x9 footprint resulting in a net
    effect that is the same as ``footprint=np.ones((9, 9))``, but with lower
    computational cost. Most of the builtin footprints such as
    ``skimage.morphology.disk`` provide an option to automatically generate a
    footprint sequence of this type.

    Examples
    --------
    >>> # Erosion shrinks bright regions
    >>> import numpy as np
    >>> from skimage.morphology import square
    >>> bright_square = np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> erosion(bright_square, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """
    if out is None:
        out = np.empty_like(image)

    if _footprint_is_sequence(footprint):
        footprints = tuple((_shift_footprint(fp, shift_x, shift_y), n)
                           for fp, n in footprint)
        return _iterate_gray_func(ndi.grey_erosion, image, footprints, out)

    footprint = np.array(footprint)
    footprint = _shift_footprint(footprint, shift_x, shift_y)
    ndi.grey_erosion(image, footprint=footprint, output=out)
    return out


@default_footprint
def dilation(image, footprint=None, out=None, shift_x=False, shift_y=False):
    """Return grayscale morphological dilation of an image.

    Morphological dilation sets the value of a pixel to the maximum over all
    pixel values within a local neighborhood centered about it. The values
    where the footprint is 1 define this neighborhood.
    Dilation enlarges bright regions and shrinks dark regions.

    Parameters
    ----------
    image : ndarray
        Image array.
    footprint : ndarray or tuple, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped footprint (connectivity=1). The footprint
        can also be provided as a sequence of smaller footprints as described
        in the notes below.
    out : ndarray, optional
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    shift_x, shift_y : bool, optional
        Shift footprint about center point. This only affects 2D
        eccentric footprints (i.e., footprints with even-numbered
        sides).

    Returns
    -------
    dilated : uint8 array, same shape and type as `image`
        The result of the morphological dilation.

    Notes
    -----
    For `uint8` (and `uint16` up to a certain bit-depth) data, the lower
    algorithm complexity makes the `skimage.filters.rank.maximum` function more
    efficient for larger images and footprints.

    The footprint can also be a provided as a sequence of 2-tuples where the
    first element of each 2-tuple is a footprint ndarray and the second element
    is an integer describing the number of times it should be iterated. For
    example ``footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]``
    would apply a 9x1 footprint followed by a 1x9 footprint resulting in a net
    effect that is the same as ``footprint=np.ones((9, 9))``, but with lower
    computational cost. Most of the builtin footprints such as
    ``skimage.morphology.disk`` provide an option to automatically generate a
    footprint sequence of this type.

    Examples
    --------
    >>> # Dilation enlarges bright regions
    >>> import numpy as np
    >>> from skimage.morphology import square
    >>> bright_pixel = np.array([[0, 0, 0, 0, 0],
    ...                          [0, 0, 0, 0, 0],
    ...                          [0, 0, 1, 0, 0],
    ...                          [0, 0, 0, 0, 0],
    ...                          [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> dilation(bright_pixel, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """
    if out is None:
        out = np.empty_like(image)

    if _footprint_is_sequence(footprint):
        # shift and invert (see comment below) each footprint
        footprints = tuple(
            (_invert_footprint(_shift_footprint(fp, shift_x, shift_y)), n)
            for fp, n in footprint
        )
        return _iterate_gray_func(ndi.grey_dilation, image, footprints, out)

    footprint = np.array(footprint)
    footprint = _shift_footprint(footprint, shift_x, shift_y)
    # Inside ndi.grey_dilation, the footprint is inverted,
    # e.g. `footprint = footprint[::-1, ::-1]` for 2D [1]_, for reasons unknown
    # to this author (@jni). To "patch" this behaviour, we invert our own
    # footprint before passing it to `ndi.grey_dilation`.
    # [1] https://github.com/scipy/scipy/blob/ec20ababa400e39ac3ffc9148c01ef86d5349332/scipy/ndimage/morphology.py#L1285  # noqa
    footprint = _invert_footprint(footprint)

    ndi.grey_dilation(image, footprint=footprint, output=out)
    return out


@default_footprint
@pad_for_eccentric_footprints
def opening(image, footprint=None, out=None):
    """Return grayscale morphological opening of an image.

    The morphological opening of an image is defined as an erosion followed by
    a dilation. Opening can remove small bright spots (i.e. "salt") and connect
    small dark cracks. This tends to "open" up (dark) gaps between (bright)
    features.

    Parameters
    ----------
    image : ndarray
        Image array.
    footprint : ndarray or tuple, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped footprint (connectivity=1). The footprint
        can also be provided as a sequence of smaller footprints as described
        in the notes below.
    out : ndarray, optional
        The array to store the result of the morphology. If None
        is passed, a new array will be allocated.

    Returns
    -------
    opening : array, same shape and type as `image`
        The result of the morphological opening.

    Notes
    -----
    The footprint can also be a provided as a sequence of 2-tuples where the
    first element of each 2-tuple is a footprint ndarray and the second element
    is an integer describing the number of times it should be iterated. For
    example ``footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]``
    would apply a 9x1 footprint followed by a 1x9 footprint resulting in a net
    effect that is the same as ``footprint=np.ones((9, 9))``, but with lower
    computational cost. Most of the builtin footprints such as
    ``skimage.morphology.disk`` provide an option to automatically generate a
    footprint sequence of this type.

    Examples
    --------
    >>> # Open up gap between two bright regions (but also shrink regions)
    >>> import numpy as np
    >>> from skimage.morphology import square
    >>> bad_connection = np.array([[1, 0, 0, 0, 1],
    ...                            [1, 1, 0, 1, 1],
    ...                            [1, 1, 1, 1, 1],
    ...                            [1, 1, 0, 1, 1],
    ...                            [1, 0, 0, 0, 1]], dtype=np.uint8)
    >>> opening(bad_connection, square(3))
    array([[0, 0, 0, 0, 0],
           [1, 1, 0, 1, 1],
           [1, 1, 0, 1, 1],
           [1, 1, 0, 1, 1],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """
    eroded = erosion(image, footprint)
    # note: shift_x, shift_y do nothing if footprint side length is odd
    out = dilation(eroded, footprint, out=out, shift_x=True, shift_y=True)
    return out


@default_footprint
@pad_for_eccentric_footprints
def closing(image, footprint=None, out=None):
    """Return grayscale morphological closing of an image.

    The morphological closing of an image is defined as a dilation followed by
    an erosion. Closing can remove small dark spots (i.e. "pepper") and connect
    small bright cracks. This tends to "close" up (dark) gaps between (bright)
    features.

    Parameters
    ----------
    image : ndarray
        Image array.
    footprint : ndarray or tuple, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped footprint (connectivity=1). The footprint
        can also be provided as a sequence of smaller footprints as described
        in the notes below.
    out : ndarray, optional
        The array to store the result of the morphology. If None,
        a new array will be allocated.

    Returns
    -------
    closing : array, same shape and type as `image`
        The result of the morphological closing.

    Notes
    -----
    The footprint can also be a provided as a sequence of 2-tuples where the
    first element of each 2-tuple is a footprint ndarray and the second element
    is an integer describing the number of times it should be iterated. For
    example ``footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]``
    would apply a 9x1 footprint followed by a 1x9 footprint resulting in a net
    effect that is the same as ``footprint=np.ones((9, 9))``, but with lower
    computational cost. Most of the builtin footprints such as
    ``skimage.morphology.disk`` provide an option to automatically generate a
    footprint sequence of this type.

    Examples
    --------
    >>> # Close a gap between two bright lines
    >>> import numpy as np
    >>> from skimage.morphology import square
    >>> broken_line = np.array([[0, 0, 0, 0, 0],
    ...                         [0, 0, 0, 0, 0],
    ...                         [1, 1, 0, 1, 1],
    ...                         [0, 0, 0, 0, 0],
    ...                         [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> closing(broken_line, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """
    dilated = dilation(image, footprint)
    # note: shift_x, shift_y do nothing if footprint side length is odd
    out = erosion(dilated, footprint, out=out, shift_x=True, shift_y=True)
    return out


def _white_tophat_seqence(image, footprints, out):
    """Return white top hat for a sequence of footprints.

    Like SciPy's implementation, but with ``ndi.grey_erosion`` and
    ``ndi.grey_dilation`` wrapped with ``_iterate_gray_func``.
    """
    tmp = _iterate_gray_func(ndi.grey_erosion, image, footprints, out)
    tmp = _iterate_gray_func(ndi.grey_dilation, tmp.copy(), footprints, out)
    if tmp is None:
        tmp = out
    if image.dtype == np.bool_ and tmp.dtype == np.bool_:
        np.bitwise_xor(image, tmp, out=tmp)
    else:
        np.subtract(image, tmp, out=tmp)
    return tmp


@default_footprint
def white_tophat(image, footprint=None, out=None):
    """Return white top hat of an image.

    The white top hat of an image is defined as the image minus its
    morphological opening. This operation returns the bright spots of the image
    that are smaller than the footprint.

    Parameters
    ----------
    image : ndarray
        Image array.
    footprint : ndarray or tuple, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped footprint (connectivity=1). The footprint
        can also be provided as a sequence of smaller footprints as described
        in the notes below.
    out : ndarray, optional
        The array to store the result of the morphology. If None
        is passed, a new array will be allocated.

    Returns
    -------
    out : array, same shape and type as `image`
        The result of the morphological white top hat.

    Notes
    -----
    The footprint can also be a provided as a sequence of 2-tuples where the
    first element of each 2-tuple is a footprint ndarray and the second element
    is an integer describing the number of times it should be iterated. For
    example ``footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]``
    would apply a 9x1 footprint followed by a 1x9 footprint resulting in a net
    effect that is the same as ``footprint=np.ones((9, 9))``, but with lower
    computational cost. Most of the builtin footprints such as
    ``skimage.morphology.disk`` provide an option to automatically generate a
    footprint sequence of this type.

    See Also
    --------
    black_tophat

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Top-hat_transform

    Examples
    --------
    >>> # Subtract gray background from bright peak
    >>> import numpy as np
    >>> from skimage.morphology import square
    >>> bright_on_gray = np.array([[2, 3, 3, 3, 2],
    ...                            [3, 4, 5, 4, 3],
    ...                            [3, 5, 9, 5, 3],
    ...                            [3, 4, 5, 4, 3],
    ...                            [2, 3, 3, 3, 2]], dtype=np.uint8)
    >>> white_tophat(bright_on_gray, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 5, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """
    if out is image:
        opened = opening(image, footprint)
        if np.issubdtype(opened.dtype, bool):
            np.logical_xor(out, opened, out=out)
        else:
            out -= opened
        return out
    elif out is None:
        out = np.empty_like(image)
    # promote bool to a type that allows arithmetic operations
    if isinstance(image, np.ndarray) and image.dtype == bool:
        image_ = image.view(dtype=np.uint8)
    else:
        image_ = image
    if isinstance(out, np.ndarray) and out.dtype == bool:
        out_ = out.view(dtype=np.uint8)
    else:
        out_ = out
    if _footprint_is_sequence(footprint):
        return _white_tophat_seqence(image_, footprint, out_)
    footprint = np.array(footprint)
    out_ = ndi.white_tophat(image_, footprint=footprint, output=out_)
    return out


@default_footprint
def black_tophat(image, footprint=None, out=None):
    """Return black top hat of an image.

    The black top hat of an image is defined as its morphological closing minus
    the original image. This operation returns the dark spots of the image that
    are smaller than the footprint. Note that dark spots in the
    original image are bright spots after the black top hat.

    Parameters
    ----------
    image : ndarray
        Image array.
    footprint : ndarray or tuple, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped footprint (connectivity=1). The footprint
        can also be provided as a sequence of smaller footprints as described
        in the notes below.
    out : ndarray, optional
        The array to store the result of the morphology. If None
        is passed, a new array will be allocated.

    Returns
    -------
    out : array, same shape and type as `image`
        The result of the morphological black top hat.

    Notes
    -----
    The footprint can also be a provided as a sequence of 2-tuples where the
    first element of each 2-tuple is a footprint ndarray and the second element
    is an integer describing the number of times it should be iterated. For
    example ``footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]``
    would apply a 9x1 footprint followed by a 1x9 footprint resulting in a net
    effect that is the same as ``footprint=np.ones((9, 9))``, but with lower
    computational cost. Most of the builtin footprints such as
    ``skimage.morphology.disk`` provide an option to automatically generate a
    footprint sequence of this type.

    See Also
    --------
    white_tophat

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Top-hat_transform

    Examples
    --------
    >>> # Change dark peak to bright peak and subtract background
    >>> import numpy as np
    >>> from skimage.morphology import square
    >>> dark_on_gray = np.array([[7, 6, 6, 6, 7],
    ...                          [6, 5, 4, 5, 6],
    ...                          [6, 4, 0, 4, 6],
    ...                          [6, 5, 4, 5, 6],
    ...                          [7, 6, 6, 6, 7]], dtype=np.uint8)
    >>> black_tophat(dark_on_gray, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 5, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """
    if out is image:
        original = image.copy()
    else:
        original = image
    out = closing(image, footprint, out=out)
    if np.issubdtype(out.dtype, np.bool_):
        np.logical_xor(out, original, out=out)
    else:
        out -= original
    return out
