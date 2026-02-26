"""
Grayscale morphological operations
"""

from .misc import default_footprint
from .. import PendingSkimage2Change
from .._shared._warnings import warn_external

import skimage2 as ski2


__all__ = ['erosion', 'dilation', 'opening', 'closing', 'white_tophat', 'black_tophat']


_SUPPORTED_MODES = {
    "reflect",
    "constant",
    "nearest",
    "mirror",
    "wrap",
    "max",
    "min",
    "ignore",
}


_PENDING_SKIMAGE2_MESSAGE = """\
`skimage.morphology.{name}` is deprecated in favor of
`skimage2.morphology.{name}`, which changes the default value
for parameter `mode` from 'reflect' to 'ignore'.

To keep the old (`skimage`, v1.x) behavior, set that parameter explicitly.
"""


@default_footprint
def erosion(
    image,
    footprint=None,
    out=None,
    *,
    mode="reflect",
    cval=0.0,
):
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
    out : ndarray, optional
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    mode : str, optional
        The `mode` parameter determines how the array borders are handled.
        Valid modes are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap',
        'max', 'min', or 'ignore'.
        If 'max' or 'ignore', pixels outside the image domain are assumed
        to be the maximum for the image's dtype, which causes them to not
        influence the result. Default is 'reflect'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.

        .. versionadded:: 0.23
            `mode` and `cval` were added in 0.23.

    Returns
    -------
    eroded : array, same shape as `image`
        The result of the morphological erosion.

    Notes
    -----
    For ``uint8`` (and ``uint16`` up to a certain bit-depth) data, the
    lower algorithm complexity makes the :func:`skimage.filters.rank.minimum`
    function more efficient for larger images and footprints.

    The footprint can also be a provided as a sequence of 2-tuples where the
    first element of each 2-tuple is a footprint ndarray and the second element
    is an integer describing the number of times it should be iterated. For
    example ``footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]``
    would apply a 9x1 footprint followed by a 1x9 footprint resulting in a net
    effect that is the same as ``footprint=np.ones((9, 9))``, but with lower
    computational cost. Most of the builtin footprints such as
    :func:`skimage.morphology.disk` provide an option to automatically generate
    a footprint sequence of this type.

    For even-sized footprints, :func:`skimage.morphology.binary_erosion` and
    this function produce an output that differs: one is shifted by one pixel
    compared to the other. :func:`skimage.morphology.pad_footprint` is available
    to account for this.

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
    warn_external(
        _PENDING_SKIMAGE2_MESSAGE.format(name=erosion.__name__),
        category=PendingSkimage2Change,
    )
    out = ski2.morphology.erosion(
        image, footprint=footprint, out=out, mode=mode, cval=cval
    )
    return out


@default_footprint
def dilation(
    image,
    footprint=None,
    out=None,
    *,
    mode="reflect",
    cval=0.0,
):
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
    mode : str, optional
        The `mode` parameter determines how the array borders are handled.
        Valid modes are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap',
        'max', 'min', or 'ignore'.
        If 'min' or 'ignore', pixels outside the image domain are assumed
        to be the maximum for the image's dtype, which causes them to not
        influence the result. Default is 'reflect'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.

        .. versionadded:: 0.23
            `mode` and `cval` were added in 0.23.

    Returns
    -------
    dilated : uint8 array, same shape and type as `image`
        The result of the morphological dilation.

    Notes
    -----
    For ``uint8`` (and ``uint16`` up to a certain bit-depth) data, the lower
    algorithm complexity makes the :func:`skimage.filters.rank.maximum`
    function more efficient for larger images and footprints.

    The footprint can also be a provided as a sequence of 2-tuples where the
    first element of each 2-tuple is a footprint ndarray and the second element
    is an integer describing the number of times it should be iterated. For
    example ``footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]``
    would apply a 9x1 footprint followed by a 1x9 footprint resulting in a net
    effect that is the same as ``footprint=np.ones((9, 9))``, but with lower
    computational cost. Most of the builtin footprints such as
    :func:`skimage.morphology.disk` provide an option to automatically generate
    a footprint sequence of this type.

    For non-symmetric footprints, :func:`skimage.morphology.binary_dilation`
    and :func:`skimage.morphology.dilation` produce an output that differs:
    `binary_dilation` mirrors the footprint, whereas `dilation` does not.
    :func:`skimage.morphology.mirror_footprint` is available to correct for this.

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
    warn_external(
        _PENDING_SKIMAGE2_MESSAGE.format(name=dilation.__name__),
        category=PendingSkimage2Change,
    )
    out = ski2.morphology.dilation(
        image, footprint=footprint, out=out, mode=mode, cval=cval
    )
    return out


@default_footprint
def opening(image, footprint=None, out=None, *, mode="reflect", cval=0.0):
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
    mode : str, optional
        The `mode` parameter determines how the array borders are handled.
        Valid modes are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap',
        'max', 'min', or 'ignore'.
        If 'ignore', pixels outside the image domain are assumed
        to be the maximum for the image's dtype in the erosion, and minimum
        in the dilation, which causes them to not influence the result.
        Default is 'reflect'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.

        .. versionadded:: 0.23
            `mode` and `cval` were added in 0.23.

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
    :func:`skimage.morphology.disk` provide an option to automatically generate
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
    warn_external(
        _PENDING_SKIMAGE2_MESSAGE.format(name=opening.__name__),
        category=PendingSkimage2Change,
    )
    out = ski2.morphology.opening(
        image, footprint=footprint, out=out, mode=mode, cval=cval
    )
    return out


@default_footprint
def closing(image, footprint=None, out=None, *, mode="reflect", cval=0.0):
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
    mode : str, optional
        The `mode` parameter determines how the array borders are handled.
        Valid modes are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap',
        'max', 'min', or 'ignore'.
        If 'ignore', pixels outside the image domain are assumed
        to be the maximum for the image's dtype in the erosion, and minimum
        in the dilation, which causes them to not influence the result.
        Default is 'reflect'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.

        .. versionadded:: 0.23
            `mode` and `cval` were added in 0.23.

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
    :func:`skimage.morphology.disk` provide an option to automatically generate
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
    warn_external(
        _PENDING_SKIMAGE2_MESSAGE.format(name=closing.__name__),
        category=PendingSkimage2Change,
    )
    out = ski2.morphology.closing(
        image, footprint=footprint, out=out, mode=mode, cval=cval
    )
    return out


@default_footprint
def white_tophat(image, footprint=None, out=None, *, mode="reflect", cval=0.0):
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
    mode : str, optional
        The `mode` parameter determines how the array borders are handled.
        Valid modes are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap',
        'max', 'min', or 'ignore'. See :func:`skimage.morphology.opening`.
        Default is 'reflect'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.

        .. versionadded:: 0.23
            `mode` and `cval` were added in 0.23.

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
    :func:`skimage.morphology.disk` provide an option to automatically generate
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
    warn_external(
        _PENDING_SKIMAGE2_MESSAGE.format(name=white_tophat.__name__),
        category=PendingSkimage2Change,
    )
    out = ski2.morphology.white_tophat(
        image, footprint=footprint, out=out, mode=mode, cval=cval
    )
    return out


@default_footprint
def black_tophat(image, footprint=None, out=None, *, mode="reflect", cval=0.0):
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
    mode : str, optional
        The `mode` parameter determines how the array borders are handled.
        Valid modes are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap',
        'max', 'min', or 'ignore'. See :func:`skimage.morphology.closing`.
        Default is 'reflect'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.

        .. versionadded:: 0.23
            `mode` and `cval` were added in 0.23.

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
    :func:`skimage.morphology.disk` provide an option to automatically generate
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
    warn_external(
        _PENDING_SKIMAGE2_MESSAGE.format(name=black_tophat.__name__),
        category=PendingSkimage2Change,
    )
    out = ski2.morphology.black_tophat(
        image, footprint=footprint, out=out, mode=mode, cval=cval
    )
    return out
