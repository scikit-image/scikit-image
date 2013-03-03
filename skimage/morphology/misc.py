import numpy as np
import scipy.ndimage as nd

def remove_small_connected_components(ar, min_size=64, 
                                            connectivity=1, in_place=False):
    """Remove connected components smaller than the specified size.

    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the connected components of interest.
    min_size : int, optional (default: 64)
        The smallest allowable connected component size.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel.
    in_place : bool, optional (default: False)
        If `True`, remove the connected components in the input array itself.
        Otherwise, make a copy.

    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small connected components removed.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import morphology
    >>> from scipy import ndimage as nd
    >>> a = np.array([[0, 0, 0, 1, 0],
    ...               [1, 1, 1, 0, 0],
    ...               [1, 1, 1, 0, 1]], bool)
    >>> b = morphology.remove_small_connected_components(a, 6)
    >>> b
    array([[False, False, False, False, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]], dtype=bool)
    >>> c = morphology.remove_small_connected_components(a, 7, connectivity=2)
    >>> c
    array([[False, False, False,  True, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]], dtype=bool)
    >>> d = morphology.remove_small_connected_components(a, 6, in_place=True)
    >>> d is a
    True
    """
    structuring_element = nd.generate_binary_structure(ar.ndim, connectivity)
    if in_place:
        out = ar
    else:
        out = ar.copy()
    if min_size == 0: # shortcut for efficiency
        return out
    if out.dtype == bool:
        ccs = nd.label(ar, structuring_element)[0]
    else:
        ccs = out
    component_sizes = np.bincount(ccs.ravel())
    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0
    return out
