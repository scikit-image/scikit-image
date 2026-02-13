"""Grayscale morphological operations."""

from typing import Final

import numpy as np
from scipy import ndimage as ndi

from skimage.morphology.footprints import (
    _footprint_is_sequence,
    mirror_footprint,
    pad_footprint,
)
from skimage.morphology.misc import default_footprint


def _apply_decomposed(*, operator, image, footprints, out, mode, cval):
    """Helper to call `gray_func` on a decomposed footprint.

    `operator` is a morphology function that accepts `footprint`, `output`,
    `mode`, and `cval` keyword arguments.

    Parameters
    ----------
    operator : Callable
        Morphological operator.
    image : ndarray
        The image to apply the operator to.
    footprints : tuple
        Decomposed footprint.
    out : ndarray
        The array to store the result in.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap', 'max', 'min', 'ignore'}
        Determine handling of array borders.
    cval : scalar
        Value to fill past edges of input if `mode` is 'constant'.

    Returns
    -------
    out : ndarray
        Processed `image`. Same as `out` parameter.
    """
    fp, num_iter = footprints[0]
    operator(image, footprint=fp, output=out, mode=mode, cval=cval)
    for _ in range(1, num_iter):
        # Note: out.copy() because the computation cannot be in-place!
        operator(out.copy(), footprint=fp, output=out, mode=mode, cval=cval)
    for fp, num_iter in footprints[1:]:
        # Note: out.copy() because the computation cannot be in-place!
        for _ in range(num_iter):
            operator(out.copy(), footprint=fp, output=out, mode=mode, cval=cval)
    return out


def _min_max_to_constant_mode(dtype, mode, cval):
    """Replace 'max' and 'min' with appropriate 'cval' and 'constant' mode.

    Parameters
    ----------
    dtype : dtype-like
        The dtype to choose an appropriate `cval` from.
        Ignored if `mode` is not 'min' or 'max'.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap', 'max', 'min', 'ignore'}
        Determine handling of array borders.
    cval : scalar
        Value to fill past edges of input if `mode` is 'constant'.

    Returns
    -------
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap', 'ignore'}
        Potentially replaced mode. Only 'min' or 'max' are replaced.
    cval : scalar
        The chosen constant value to fill with.
    """
    if mode == "max":
        mode = "constant"
        if np.issubdtype(dtype, bool):
            cval = True
        elif np.issubdtype(dtype, np.integer):
            cval = np.iinfo(dtype).max
        else:
            cval = np.inf
    elif mode == "min":
        mode = "constant"
        if np.issubdtype(dtype, bool):
            cval = False
        elif np.issubdtype(dtype, np.integer):
            cval = np.iinfo(dtype).min
        else:
            cval = -np.inf
    return mode, cval


_SUPPORTED_MODES: Final = {
    "reflect",
    "constant",
    "nearest",
    "mirror",
    "wrap",
    "max",
    "min",
    "ignore",
}


@default_footprint
def erosion(image, footprint=None, *, out=None, mode="ignore", cval=0.0):
    """Return grayscale morphological erosion of an image.

    Morphological erosion shrinks bright regions and enlarges dark regions.
    It sets a pixel at ``(i, j)`` to the minimum over all pixels in the neighborhood
    centered at ``(i, j)``. The values where the footprint is 1 define this
    neighborhood.

    Parameters
    ----------
    image : ndarray
        Input image.
    footprint : ndarray or tuple, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped footprint (so-called 1-connectivity). The footprint
        can also be provided as a sequence of smaller footprints as described
        in the notes below.
    out : ndarray, optional
        The array to store the result of the morphology. If None,
        a new array is allocated.
    mode : str, optional
        The `mode` parameter determines how the array borders are handled.
        Valid modes are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap',
        'max', 'min', or 'ignore'.
        If 'max' or 'ignore', pixels outside the image domain are assumed
        to be the maximum for the image's dtype, which causes them to not
        influence the result. Default is 'ignore'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.

    Returns
    -------
    out : ndarray, same shape and dtype as `image`
        The result of the morphological erosion.

    Notes
    -----
    For ``uint8`` (and ``uint16`` up to a certain bit-depth) data, the
    lower algorithm complexity makes the :func:`skimage2.filters.rank.minimum`
    function more efficient for larger images and footprints.

    The footprint can also be provided as a sequence of 2-tuples where the
    first element of each 2-tuple is a footprint ndarray and the second element
    is an integer describing the number of times it should be iterated. For
    example, ``footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]``
    would apply a 9x1 footprint followed by a 1x9 footprint resulting in a net
    effect that is the same as ``footprint=np.ones((9, 9))``, but with lower
    computational cost. Most of the built-in footprints such as
    :func:`skimage2.morphology.disk` provide an option to automatically generate
    a footprint sequence of this type.

    Examples
    --------
    >>> # Erosion shrinks bright regions
    >>> import numpy as np
    >>> from skimage.morphology import footprint_rectangle
    >>> bright_square = np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> erosion(bright_square, footprint_rectangle((3, 3)))
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """
    if out is None:
        out = np.empty_like(image)

    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"unsupported mode, got {mode!r}")
    if mode == "ignore":
        mode = "max"
    mode, cval = _min_max_to_constant_mode(image.dtype, mode, cval)

    footprint = pad_footprint(footprint, pad_end=False)
    if not _footprint_is_sequence(footprint):
        footprint = [(footprint, 1)]

    out = _apply_decomposed(
        operator=ndi.grey_erosion,
        image=image,
        footprints=footprint,
        out=out,
        mode=mode,
        cval=cval,
    )
    return out


@default_footprint
def dilation(image, footprint=None, *, out=None, mode="ignore", cval=0.0):
    """Return grayscale morphological dilation of an image.

    Morphological dilation enlarges bright regions and shrinks dark regions.
    It sets a pixel at ``(i, j)`` to the maximum over all pixels in the neighborhood
    centered at ``(i, j)``. The values where the footprint is 1 define this
    neighborhood.

    Parameters
    ----------
    image : ndarray
        Input image.
    footprint : ndarray or tuple, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped footprint (so-called 1-connectivity). The footprint
        can also be provided as a sequence of smaller footprints as described
        in the notes below.
    out : ndarray, optional
        The array to store the result of the morphology. If None,
        a new array is allocated.
    mode : str, optional
        The `mode` parameter determines how the array borders are handled.
        Valid modes are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap',
        'max', 'min', or 'ignore'.
        If 'min' or 'ignore', pixels outside the image domain are assumed
        to be the maximum for the image's dtype, which causes them to not
        influence the result. Default is 'ignore'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.

    Returns
    -------
    out : ndarray, same shape and dtype as `image`
        The result of the morphological dilation.

    Notes
    -----
    For ``uint8`` (and ``uint16`` up to a certain bit-depth) data, the lower
    algorithm complexity makes the :func:`skimage2.filters.rank.maximum`
    function more efficient for larger images and footprints.

    The footprint can also be a provided as a sequence of 2-tuples where the
    first element of each 2-tuple is a footprint ndarray and the second element
    is an integer describing the number of times it should be iterated. For
    example ``footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]``
    would apply a 9x1 footprint followed by a 1x9 footprint resulting in a net
    effect that is the same as ``footprint=np.ones((9, 9))``, but with lower
    computational cost. Most of the builtin footprints such as
    :func:`skimage2.morphology.disk` provide an option to automatically generate
    a footprint sequence of this type.

    Examples
    --------
    >>> # Dilation enlarges bright regions
    >>> import numpy as np
    >>> from skimage.morphology import footprint_rectangle
    >>> bright_pixel = np.array([[0, 0, 0, 0, 0],
    ...                          [0, 0, 0, 0, 0],
    ...                          [0, 0, 1, 0, 0],
    ...                          [0, 0, 0, 0, 0],
    ...                          [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> dilation(bright_pixel, footprint_rectangle((3, 3)))
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """
    if out is None:
        out = np.empty_like(image)

    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"unsupported mode, got {mode!r}")
    if mode == "ignore":
        mode = "min"
    mode, cval = _min_max_to_constant_mode(image.dtype, mode, cval)

    footprint = pad_footprint(footprint, pad_end=False)
    # Note that `ndi.grey_dilation` mirrors the footprint and this
    # additional inversion should be removed in skimage2, see gh-6676.
    footprint = mirror_footprint(footprint)
    if not _footprint_is_sequence(footprint):
        footprint = [(footprint, 1)]

    out = _apply_decomposed(
        operator=ndi.grey_dilation,
        image=image,
        footprints=footprint,
        out=out,
        mode=mode,
        cval=cval,
    )
    return out


@default_footprint
def opening(image, footprint=None, *, out=None, mode="ignore", cval=0.0):
    """Return grayscale morphological opening of an image.

    The morphological opening of an image is defined as an erosion followed by
    a dilation. Opening can remove small bright spots (i.e., "salt") and connect
    small dark cracks. This tends to "open" up (dark) gaps between (bright)
    features.

    Parameters
    ----------
    image : ndarray
        Input image.
    footprint : ndarray or tuple, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped footprint (so-called 1-connectivity). The footprint
        can also be provided as a sequence of smaller footprints as described
        in the notes below.
    out : ndarray, optional
        The array to store the result of the morphology. If None,
        a new array is allocated.
    mode : str, optional
        The `mode` parameter determines how the array borders are handled.
        Valid modes are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap',
        'max', 'min', or 'ignore'.
        If 'ignore', pixels outside the image domain are assumed
        to be the maximum for the image's dtype in the erosion, and minimum
        in the dilation, which causes them to not influence the result.
        Default is 'ignore'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.

    Returns
    -------
    out : ndarray, same shape and dtype as `image`
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
    :func:`skimage2.morphology.disk` provide an option to automatically generate
    a footprint sequence of this type.

    Examples
    --------
    >>> # Open up gap between two bright regions (but also shrink regions)
    >>> import numpy as np
    >>> from skimage.morphology import footprint_rectangle
    >>> bad_connection = np.array([[1, 0, 0, 0, 1],
    ...                            [1, 1, 0, 1, 1],
    ...                            [1, 1, 1, 1, 1],
    ...                            [1, 1, 0, 1, 1],
    ...                            [1, 0, 0, 0, 1]], dtype=np.uint8)
    >>> opening(bad_connection, footprint_rectangle((3, 3)))
    array([[0, 0, 0, 0, 0],
           [1, 1, 0, 1, 1],
           [1, 1, 0, 1, 1],
           [1, 1, 0, 1, 1],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """
    footprint = pad_footprint(footprint, pad_end=False)
    eroded = erosion(image, footprint, mode=mode, cval=cval)
    out = dilation(eroded, mirror_footprint(footprint), out=out, mode=mode, cval=cval)
    return out


@default_footprint
def closing(image, footprint=None, *, out=None, mode="ignore", cval=0.0):
    """Return grayscale morphological closing of an image.

    The morphological closing of an image is defined as a dilation followed by
    an erosion. Closing can remove small dark spots (i.e., "pepper") and connect
    small bright cracks. This tends to "close" up (dark) gaps between (bright)
    features.

    Parameters
    ----------
    image : ndarray
        Input image.
    footprint : ndarray or tuple, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped footprint (so-called 1-connectivity). The footprint
        can also be provided as a sequence of smaller footprints as described
        in the notes below.
    out : ndarray, optional
        The array to store the result of the morphology. If None,
        a new array is allocated.
    mode : str, optional
        The `mode` parameter determines how the array borders are handled.
        Valid modes are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap',
        'max', 'min', or 'ignore'.
        If 'ignore', pixels outside the image domain are assumed
        to be the maximum for the image's dtype in the erosion, and minimum
        in the dilation, which causes them to not influence the result.
        Default is 'ignore'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.

    Returns
    -------
    out : ndarray, same shape and dtype as `image`
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
    :func:`skimage2.morphology.disk` provide an option to automatically generate
    a footprint sequence of this type.

    Examples
    --------
    >>> # Close a gap between two bright lines
    >>> import numpy as np
    >>> from skimage.morphology import footprint_rectangle
    >>> broken_line = np.array([[0, 0, 0, 0, 0],
    ...                         [0, 0, 0, 0, 0],
    ...                         [1, 1, 0, 1, 1],
    ...                         [0, 0, 0, 0, 0],
    ...                         [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> closing(broken_line, footprint_rectangle((3, 3)))
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """
    footprint = pad_footprint(footprint, pad_end=False)
    dilated = dilation(image, footprint, mode=mode, cval=cval)
    out = erosion(dilated, mirror_footprint(footprint), out=out, mode=mode, cval=cval)
    return out


@default_footprint
def white_tophat(image, footprint=None, *, out=None, mode="ignore", cval=0.0):
    """Return white top hat of an image.

    The white top hat of an image is defined as the image minus its
    morphological opening. This operation returns the bright spots of the image
    that are smaller than the footprint.

    Parameters
    ----------
    image : ndarray
        Input image.
    footprint : ndarray or tuple, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped footprint (so-called 1-connectivity). The footprint
        can also be provided as a sequence of smaller footprints as described
        in the notes below.
    out : ndarray, optional
        The array to store the result of the morphology. If None,
        a new array is allocated.
    mode : str, optional
        The `mode` parameter determines how the array borders are handled.
        Valid modes are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap',
        'max', 'min', or 'ignore'. See :func:`skimage2.morphology.opening`.
        Default is 'ignore'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.

    Returns
    -------
    out : ndarray, same shape and dtype as `image`
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
    :func:`skimage2.morphology.disk` provide an option to automatically generate
    a footprint sequence of this type.

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
    >>> from skimage.morphology import footprint_rectangle
    >>> bright_on_gray = np.array([[2, 3, 3, 3, 2],
    ...                            [3, 4, 5, 4, 3],
    ...                            [3, 5, 9, 5, 3],
    ...                            [3, 4, 5, 4, 3],
    ...                            [2, 3, 3, 3, 2]], dtype=np.uint8)
    >>> white_tophat(bright_on_gray, footprint_rectangle((3, 3)))
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 5, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)
    """
    if out is image:
        # We need a temporary image
        opened = opening(image, footprint, mode=mode, cval=cval)
        if np.issubdtype(opened.dtype, bool):
            np.logical_xor(out, opened, out=out)
        else:
            out -= opened
        return out

    # Else write intermediate result into output image
    out = opening(image, footprint, out=out, mode=mode, cval=cval)
    if np.issubdtype(out.dtype, bool):
        np.logical_xor(image, out, out=out)
    else:
        np.subtract(image, out, out=out)
    return out


@default_footprint
def black_tophat(image, footprint=None, *, out=None, mode="ignore", cval=0.0):
    """Return black top hat of an image.

    The black top hat of an image is defined as its morphological closing minus
    the original image. This operation returns the dark spots of the image that
    are smaller than the footprint. Note that dark spots in the
    original image are bright spots after the black top hat.

    Parameters
    ----------
    image : ndarray
        Input image.
    footprint : ndarray or tuple, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped footprint (so-called 1-connectivity). The footprint
        can also be provided as a sequence of smaller footprints as described
        in the notes below.
    out : ndarray, optional
        The array to store the result of the morphology. If None,
        a new array is allocated.
    mode : str, optional
        The `mode` parameter determines how the array borders are handled.
        Valid modes are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap',
        'max', 'min', or 'ignore'. See :func:`skimage2.morphology.closing`.
        Default is 'ignore'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.

    Returns
    -------
    out : ndarray, same shape and dtype as `image`
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
    :func:`skimage2.morphology.disk` provide an option to automatically generate
    a footprint sequence of this type.

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
    >>> from skimage.morphology import footprint_rectangle
    >>> dark_on_gray = np.array([[7, 6, 6, 6, 7],
    ...                          [6, 5, 4, 5, 6],
    ...                          [6, 4, 0, 4, 6],
    ...                          [6, 5, 4, 5, 6],
    ...                          [7, 6, 6, 6, 7]], dtype=np.uint8)
    >>> black_tophat(dark_on_gray, footprint_rectangle((3, 3)))
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 5, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """
    if out is image:
        # We need a temporary image
        closed = closing(image, footprint, mode=mode, cval=cval)
        if np.issubdtype(closed.dtype, bool):
            np.logical_xor(closed, out, out=out)
        else:
            np.subtract(closed, out, out=out)
        return out

    out = closing(image, footprint, out=out, mode=mode, cval=cval)
    if np.issubdtype(out.dtype, np.bool_):
        np.logical_xor(out, image, out=out)
    else:
        out -= image
    return out
