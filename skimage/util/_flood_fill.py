"""flood_fill.py - inplace flood fill algorithm

This module provides a function to fill all equal (or within tolerance) values
connected to a given seed point with a different value.
"""

import numpy as np


def flood_fill(image, seed_point, new_value, *, selem=None, connectivity=None,
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
        If True, the output will be an array representing indices of the flood
        fill. If False (default), the output will be an n-dimensional array
        with flood filling applied.
    tolerance : ``selem`` type, optional
        If None (default), adjacent values must be strictly equal to the
        initial value of `image` at `seed_point`.  This is fastest.  If a value
        is given, a comparison will be done at every point and this tolerance
        on each side of the initial value will also be filled (inclusive).
    inplace : bool, optional
        If True, flood filling is applied to `image` inplace.  If False, the
        flood filled result is returned without modifying the input `image`
        (default).  Ignored if `indices` is True.

    Returns
    -------
    filled : ndarray or tuple[ndarray]
        If `indices` is false, an array with the same shape as `image` is
        returned with values equal to (or within tolerance of) the seed point
        set to `new_value`.  If `indices` is true, a tuple of one-dimensional
        arrays containing the coordinates (indices) of all found maxima is
        returned.

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
    from ..morphology.extrema import (_resolve_neighborhood,
                                      _set_edge_values_inplace, _fast_pad)
    from ..morphology.watershed import _offsets_to_raveled_neighbors
    from ._flood_fill_cy import _flood_fill_equal, _flood_fill_tolerance

    image = np.asarray(image)
    seed_value = image[seed_point]

    # Shortcut for rank zero
    if image.size == 0:
        return np.array([], dtype=(np.intp if indices else np.uint8))

    # Convenience for 1d input
    try:
        iter(seed_point)
    except TypeError:
        seed_point = (seed_point,)

    selem = _resolve_neighborhood(selem, connectivity, image.ndim)

    # Must annotate borders
    working_image = _fast_pad(image, image.min())

    # Correct start point in ravelled image
    ravelled_seed_idx = np.ravel_multi_index([i+1 for i in seed_point],
                                             working_image.shape)
    neighbor_offsets = _offsets_to_raveled_neighbors(
        working_image.shape, selem, center=((1,) * image.ndim))

    # Use a set of flags; see _flood_fill_cy.pyx for meanings
    flags = np.zeros(working_image.shape, dtype=np.uint8)
    _set_edge_values_inplace(flags, value=2)

    try:
        if tolerance is not None:
            # Check if tolerance could create overflow problems
            try:
                max_value = np.finfo(working_image.dtype).max
                min_value = np.finfo(working_image.dtype).min
            except ValueError:
                max_value = np.iinfo(working_image.dtype).max
                min_value = np.iinfo(working_image.dtype).min

            high_tol = min(max_value, seed_value + tolerance)
            low_tol = max(min_value, seed_value - tolerance)

            _flood_fill_tolerance(working_image.ravel(),
                                  flags.ravel(),
                                  neighbor_offsets,
                                  ravelled_seed_idx,
                                  seed_value,
                                  low_tol,
                                  high_tol)
        else:
            _flood_fill_equal(working_image.ravel(),
                              flags.ravel(),
                              neighbor_offsets,
                              ravelled_seed_idx,
                              seed_value)
    except TypeError:
        if working_image.dtype == np.float16:
            # Provide the user with clearer error message
            raise TypeError("dtype of `image` is float16 which is not "
                            "supported, try upcasting to float32")
        else:
            raise

    # Output what the user requested
    original_slice = (slice(1, -1),) * image.ndim
    if indices:
        return np.nonzero(flags[original_slice] == 1)
    else:
        if not inplace:
            output = working_image[original_slice]
            output[flags[original_slice] == 1] = new_value
            return output
        else:
            image[flags[original_slice] == 1] = new_value
            return image
