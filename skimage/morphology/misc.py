import numpy as np
import scipy.ndimage as nd
from .selem import _default_selem

# Our function names don't exactly correspond to ndimages.
# This dictionary translates from our names to scipy's.
funcs = ('erosion', 'dilation', 'opening', 'closing')
skimage2ndimage = dict((x, 'grey_' + x) for x in funcs)

# These function names are the same in ndimage.
funcs = ('binary_erosion', 'binary_dilation', 'binary_opening',
         'binary_closing', 'black_tophat', 'white_tophat')
skimage2ndimage.update(dict((x, x) for x in funcs))


def default_fallback(func):
    """Decorator to fall back on ndimage for images with more than 2 dimensions

    Decorator also provides a default structuring element, `selem`, with the
    appropriate dimensionality if none is specified.

    Parameters
    ----------
    func : function
        A morphology function such as erosion, dilation, opening, closing,
        white_tophat, or black_tophat.

    Returns
    -------
    func_out : function
        If the image dimentionality is greater than 2D, the ndimage
        function is returned, otherwise skimage function is used.
    """

    def func_out(image, selem=None, out=None, **kwargs):
        # Default structure element
        if selem is None:
            selem = _default_selem(image.ndim)

        # If image has more than 2 dimensions, use scipy.ndimage
        if image.ndim > 2:
            function = getattr(nd, skimage2ndimage[func.__name__])
            try:
                return function(image, footprint=selem, output=out, **kwargs)
            except TypeError:
                # nd.binary_* take structure instead of footprint
                return function(image, structure=selem, output=out, **kwargs)
        else:
            return func(image, selem=selem, out=out, **kwargs)

    return func_out


def remove_small_objects(ar, min_size=64, connectivity=1, in_place=False):
    """Remove connected components smaller than the specified size.

    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the connected components of interest. If the array
        type is int, it is assumed that it contains already-labeled objects.
        The ints must be non-negative.
    min_size : int, optional (default: 64)
        The smallest allowable connected component size.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel.
    in_place : bool, optional (default: False)
        If `True`, remove the connected components in the input array itself.
        Otherwise, make a copy.

    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or string.
    ValueError
        If the input array contains negative values.

    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small connected components removed.

    Examples
    --------
    >>> from skimage import morphology
    >>> a = np.array([[0, 0, 0, 1, 0],
    ...               [1, 1, 1, 0, 0],
    ...               [1, 1, 1, 0, 1]], bool)
    >>> b = morphology.remove_small_objects(a, 6)
    >>> b
    array([[False, False, False, False, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]], dtype=bool)
    >>> c = morphology.remove_small_objects(a, 7, connectivity=2)
    >>> c
    array([[False, False, False,  True, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]], dtype=bool)
    >>> d = morphology.remove_small_objects(a, 6, in_place=True)
    >>> d is a
    True
    """
    # Should use `issubdtype` for bool below, but there's a bug in numpy 1.7
    if not (ar.dtype == bool or np.issubdtype(ar.dtype, np.integer)):
        raise TypeError("Only bool or integer image types are supported. "
                        "Got %s." % ar.dtype)

    if in_place:
        out = ar
    else:
        out = ar.copy()

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = nd.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        nd.label(ar, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out
