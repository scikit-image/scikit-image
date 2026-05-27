from warnings import warn

import numpy as np
from scipy import ndimage as ndi

from .rank import generic


def median(
    image, footprint=None, out=None, mode='nearest', cval=0.0, behavior='ndimage'
):
    """Return local median of an image.

    Parameters
    ----------
    image : array-like
        Input image.
    footprint : ndarray, optional
        If ``behavior='rank'``, `footprint` is a 2-D array of 1's and 0's.
        If ``behavior='ndimage'``, `footprint` is a N-D array of 1's and 0's
        with the same number of dimensions as `image`.
        If None, `footprint` will be a N-D array with 3 elements for each
        dimension (e.g., vector, square, cube, etc.).
    out : ndarray, optional
        If None, a new array is allocated. For ``behavior='ndimage'``,
        dtype matches input. For ``behavior='rank'``, dtype follows
        :func:`skimage.filters.rank.median`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        `cval` is the value when `mode` is equal to 'constant'.
        Default is 'nearest'.

        .. versionadded:: 0.15
           `mode` is used when ``behavior='ndimage'``.
    cval : scalar, optional
        Value to fill past edges of input `image` if `mode` is 'constant'.
        Default is 0.0.

        .. versionadded:: 0.15
           `cval` was added in 0.15 and is used when ``behavior='ndimage'``.
    behavior : {'ndimage', 'rank'}, optional
        Whether to use the old (i.e., scikit-image < 0.15) or the new behavior.
        The old behavior will call :func:`skimage.filters.rank.median`.
        The new behavior will call :func:`scipy.ndimage.median_filter`.
        Default is 'ndimage'.

        .. versionadded:: 0.15
           `behavior` is introduced in v0.15.
        .. versionchanged:: 0.16
           Default `behavior` has been changed from 'rank' to 'ndimage'.

    Returns
    -------
    out : ndarray
        Output image. For ``behavior='ndimage'``, dtype matches input. For
        ``behavior='rank'``, dtype follows :func:`skimage.filters.rank.median`.

    See also
    --------
    skimage.filters.rank.median : Rank-based implementation of the median
        filtering offering more flexibility with additional parameters but
        dedicated for unsigned integer images.

    Examples
    --------
    >>> import skimage as ski
    >>> img = ski.data.camera()
    >>> med = ski.filters.median(img, ski.morphology.disk(5))

    """
    if behavior == 'rank':
        if mode != 'nearest' or not np.isclose(cval, 0.0):
            warn(
                "Change 'behavior' to 'ndimage' if you want to use the "
                "parameters 'mode' or 'cval'. They will be discarded "
                "otherwise.",
                stacklevel=2,
            )
        return generic.median(image, footprint=footprint, out=out)
    if footprint is None:
        footprint = ndi.generate_binary_structure(image.ndim, image.ndim)
    return ndi.median_filter(
        image, footprint=footprint, output=out, mode=mode, cval=cval
    )
