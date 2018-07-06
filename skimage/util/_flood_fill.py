"""flood_fill.py - inplace flood fill algorithm

This module provides a function to fill all equal (or within tolerance) values
connected to a given seed point with a different value.
"""

import numpy as np


def flood_fill(image, seed_point, new_value, selem=None, connectivity=None,
               indices=False, tolerance=None, inplace=False):
    """Perform flood filling on an image.

    Starting at a specific `seed_point`, connected points equal or within
    `tolerance` of the seed value are found, then set to `new_value`.

    Parameters
    ----------
    image : ndarray
        An n-dimensional array.
    seed_point : tuple or int
        The index into `image` to start filling.
    new_value : `image` type
        New value to set the entire fill.  This must be chosen in agreement
        with the dtype of `image`.
    selem : ndarray, optional
        A structuring element used to determine the neighborhood of each
        evaluated pixel. It must contain only 1's and 0's, have the same number
        of dimensions as `image`. If not given, all adjacent pixels are
        considered as part of the neighborhood (fully connected).
    connectivity : int, optional
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is larger or
        equal to `connectivity` are considered neighbors. Ignored if
        `selem` is not None.
    indices : bool, optional
        If True, the output will be an array representing indices of local
        maxima. If False, the output will be an array of 0's and 1's with the
        same shape as `image`.
    tolerance : ``selem`` type, optional
        If None (default), adjacent values must be strictly equal to the
        initial value of `image` at `seed_point`.  This is fastest.  If a value
        is given, a comparison will be done at every point and this tolerance
        on each side of the initial value will also be filled (inclusive).
    inplace : bool, optional
        If True, flood filling is applied to `image` inplace and nothing is
        returned.  If False, the flood filled result is returned without
        modifying the input `image`.  Ignored if `indices` is True.

    Returns
    -------
    filled : ndarray or tuple[ndarray]
        If `indices` is false, an array with the same shape as `image` is
        returned with values equal to (or within tolerance) of the seed point
        set to `new_value`.  If `indices` is true, a tuple of one-dimensional
        arrays containing the coordinates (indices) of all found maxima.

    See Also
    --------
    skimage.morphology.local_maxima
    skimage.morphology.local_minima
    skimage.morphology.h_maxima
    skimage.morphology.h_minima

    Notes
    -----
    The conceptual analogy of this operation is the 'paint bucket' tool in many
    raster graphics programs.

    Examples
    --------
    >>> from skimage.util import flood_fill
    >>> image = np.zeros((4, 7), dtype=int)
    >>> image[1:3, 1:3] = 1
    >>> image[3, 0] = 1
    >>> image[1:3, 4:6] = 2
    >>> image[3, 6] = 3
    >>> image
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, with full connectivity (diagonals included):

    >>> flood_fill(image, (1, 1), 5)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [5, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, with only cardinal direction connectivity:

    >>> flood_fill(image, (1, 1), 5, connectivity=1)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill with a tolerance:

    >>> flood_fill(image, (0, 0), 5, tolerance=1)
    array([[5, 5, 5, 5, 5, 5, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 5, 5, 3]])
    """
    from ..morphology.extrema import _resolve_neighborhood
    from ..morphology.watershed import _offsets_to_raveled_neighbors
    from ._flood_fill_cy import _flood_fill

    image = np.asarray(image, order='C')

    # Shortcut for rank zero
    if image.size == 0:
        return np.array([], dtype=(np.intp if indices else np.uint8))

    # Convenience for 1d input
    try:
        iter(seed_point)
    except TypeError:
        seed_point = (seed_point,)

    selem = _resolve_neighborhood(selem, connectivity, image.ndim)
    neighbor_offsets = _offsets_to_raveled_neighbors(
        image.shape, selem, center=((1,) * image.ndim))

    # Correct start point in ravelled image
    ravelled_seed_idx = np.ravel_multi_index(seed_point, image.shape)
    seed_value = image[seed_point]

    # Use a set of flags; see _flood_fil.pyx for meanings
    flags = np.zeros(image.shape, dtype=np.uint8)

    # Inform the Cython function which path to take
    if tolerance is not None:
        do_tol = 1
        # Check if tolerance could create overflow problems
        try:
            max_value = np.finfo(image.dtype).max
            min_value = np.finfo(image.dtype).min
        except ValueError:
            max_value = np.iinfo(image.dtype).max
            min_value = np.iinfo(image.dtype).min

        high_tol = min(max_value, seed_value + tolerance)
        low_tol = max(min_value, seed_value - tolerance)

    else:
        do_tol = 0
        high_tol = seed_value
        low_tol = seed_value

    # Run flood fill
    try:
        _flood_fill(image.ravel(), flags.ravel(), neighbor_offsets,
                    ravelled_seed_idx, seed_value, do_tol, high_tol, low_tol)
    except TypeError:
        if image.dtype == np.float16:
            # Provide the user with clearer error message
            raise TypeError("dtype of `image` is float16 which is not "
                            "supported, try upcasting to float32")
        else:
            raise

    if indices:
        return np.nonzero(flags == 1)
    else:
        if not inplace:
            output = image.copy()
            output[flags == 1] = new_value
            return output
        else:
            image[flags == 1] = new_value
            return
