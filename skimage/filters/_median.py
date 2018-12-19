
from warnings import warn

import numpy as np
from scipy import ndimage as ndi

from .rank import generic


@generic._default_selem
def median(image, selem=None, out=None, mask=None, shift_x=False,
           shift_y=False, mode='nearest', cval=0.0, behavior='old'):
    """Return local median of an image.

    Parameters
    ----------
    image : array-like
        Input image.
    selem : ndarray
        If ``behavior=='old'``, ``selem`` is a 2-D array of 1's and 0's.
        If ``behavior=='new'``, ``selem`` is a N-D array of 1's and 0's with
        the same number of dimension than ``image``.
        If None, ``selem`` will be a N-D array with 3 elements for each
        dimension (e.g., vector, square, cube, etc.)
    out : ndarray, (same dtype as image)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default). Only valid
        when ``behavior='old'``
        .. deprecated:: 0.17
           ``mask`` is deprecated in 0.17 and will be removed 0.19.
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element). Only valid when ``behavior='new'``.
        .. deprecated:: 0.17
           ``shift_x`` and ``shift_y`` are deprecated in 0.17 and will be
           removed in 0.19.
    mode : {'reflect', 'constant', 'nearest', 'mirror','‘wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        ``cval`` is the value when mode is equal to 'constant'.
        Default is 'nearest'.
        .. versionadded:: 0.15
           ``mode`` is used when ``behavior='new'``.
    cval : scalar, optional
        Value to fill past edges of input if mode is 'constant'. Default is 0.0
        .. versionadded:: 0.15
           ``cval`` was added in 0.15is used when ``behavior='new'``.
    behavior : {'old', 'new'}, optional
        Either to use the old behavior (i.e., < 0.15) or the new behavior.
        The new behavior will call the :func:`scipy.ndimage.median_filter`.
        Default is 'old'.
        .. versionadded:: 0.15
           ``behavior`` is introduced in 0.15
        .. deprecated:: 0.15
           Default ``behavior`` will change from 'old' to 'new' in 0.17
    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filters import median
    >>> img = data.camera()
    >>> med = median(img, disk(5))

    """
    if behavior == 'old':
        warn("Default 'behavior' will change from 'old' to 'new' in 0.17",
             DeprecationWarning)
        if mode != 'nearest' or not np.isclose(cval, 0.0):
            warn("Change 'behavior' to 'new' if you want to use the "
                 "parameters 'mode' or 'cval'. They will be discarded "
                 "otherwise.")
        return generic.median(image, selem=selem, out=out, mask=mask,
                              shift_x=shift_x, shift_y=shift_y)
    if mask is not None or shift_x or shift_y:
        warn("Change 'behavior' to 'old' if you want to use the "
             "parameters 'mask', 'shift_x', 'shift_y'. They will be "
             "discarded otherwise.")
    return ndi.median_filter(image, footprint=selem, output=out, mode=mode,
                             cval=cval)
