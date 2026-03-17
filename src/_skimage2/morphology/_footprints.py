"""Footprint generation and manipulation."""

import functools
from collections.abc import Sequence
from numbers import Integral

import numpy as np
import scipy as sp


def _default_footprint(func):
    """Decorator to add a default footprint to morphology functions.

    Parameters
    ----------
    func : Callable
        A morphology function such as erosion, dilation, opening, closing,
        white_tophat, or black_tophat.

    Returns
    -------
    func_out : Callable
        The function, using a default footprint of same dimension
        as the input image with connectivity 1.

    """

    @functools.wraps(func)
    def func_out(image, footprint=None, *args, **kwargs):
        if footprint is None:
            footprint = sp.ndimage.generate_binary_structure(image.ndim, 1)
        return func(image, footprint=footprint, *args, **kwargs)

    return func_out


def _footprint_is_sequence(footprint):
    """Check whether a footprint a (decomposed) sequence of smaller footprints.

    Parameters
    ----------
    footprint : ndarray or tuple
        The input footprint or sequence of footprints.

    Returns
    -------
    is_sequence : bool
        ``True`` if `footprint` is a valid sequence, ``False`` otherwise.

    Raises
    ------
    ValueError
        If an invalid sequence is given.

    Examples
    --------
    >>> footprint = np.ones((3, 3), dtype=bool)
    >>> _footprint_is_sequence(footprint)
    False
    >>> decomposed = ((footprint, 3),)
    >>> _footprint_is_sequence(decomposed)
    True
    """
    if hasattr(footprint, '__array_interface__'):
        return False

    def _validate_sequence_element(t):
        return (
            isinstance(t, Sequence)
            and len(t) == 2
            and hasattr(t[0], '__array_interface__')
            and isinstance(t[1], Integral)
        )

    if isinstance(footprint, Sequence):
        if len(footprint) == 0:
            raise ValueError("footprint sequence is empty")
        if not all(_validate_sequence_element(t) for t in footprint):
            raise ValueError(
                "All elements of footprint sequence must be a 2-tuple where "
                "the first element of the tuple is an ndarray and the second "
                "is an integer indicating the number of iterations."
            )
    else:
        raise ValueError("footprint must be either an ndarray or Sequence")
    return True


def mirror_footprint(footprint):
    """Mirror each dimension in the footprint.

    Parameters
    ----------
    footprint : ndarray or tuple
        The input footprint or sequence of footprints

    Returns
    -------
    inverted : ndarray or tuple
        The footprint, mirrored along each dimension.

    Examples
    --------
    >>> footprint = np.array([[0, 0, 0],
    ...                       [0, 1, 1],
    ...                       [0, 1, 1]], np.uint8)
    >>> mirror_footprint(footprint)
    array([[1, 1, 0],
           [1, 1, 0],
           [0, 0, 0]], dtype=uint8)

    """
    if _footprint_is_sequence(footprint):
        return tuple((mirror_footprint(fp), n) for fp, n in footprint)
    footprint = np.asarray(footprint)
    return footprint[(slice(None, None, -1),) * footprint.ndim]


def pad_footprint(footprint, *, pad_end=True):
    """Pad the footprint to an odd size along each dimension.

    Parameters
    ----------
    footprint : ndarray or tuple
        The input footprint or sequence of footprints
    pad_end : bool, optional
        If ``True``, pads at the end of each dimension (right side), otherwise
        pads on the front (left side).

    Returns
    -------
    padded : ndarray or tuple
        The footprint, padded to an odd size along each dimension.

    Examples
    --------
    >>> footprint = np.array([[0, 0],
    ...                       [1, 1],
    ...                       [1, 1]], np.uint8)
    >>> pad_footprint(footprint)
    array([[0, 0, 0],
           [1, 1, 0],
           [1, 1, 0]], dtype=uint8)

    """
    if _footprint_is_sequence(footprint):
        return tuple((pad_footprint(fp, pad_end=pad_end), n) for fp, n in footprint)
    footprint = np.asarray(footprint)
    padding = []
    for sz in footprint.shape:
        padding.append(((0, 1) if pad_end else (1, 0)) if sz % 2 == 0 else (0, 0))
    return np.pad(footprint, padding)
