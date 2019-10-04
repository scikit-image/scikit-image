"""Utility functions used in the morphology subpackage."""


import numpy as np
from scipy import ndimage as ndi


def _validate_connectivity(image_dim, connectivity, offset):
    """Convert any valid connectivity to a structuring element and offset.

    Parameters
    ----------
    image_dim : int
        The number of dimensions of the input image.
    connectivity : int, array, or None
        The neighborhood connectivity. An integer is interpreted as in
        ``scipy.ndimage.generate_binary_structure``, as the maximum number
        of orthogonal steps to reach a neighbor. An array is directly
        interpreted as a structuring element and its shape is validated against
        the input image shape. ``None`` is interpreted as a connectivity of 1.
    offset : tuple of int, or None
        The coordinates of the center of the structuring element.

    Returns
    -------
    c_connectivity : array of bool
        The structuring element corresponding to the input `connectivity`.
    offset : array of int
        The offset corresponding to the center of the structuring element.

    Raises
    ------
    ValueError:
        If the image dimension and the connectivity or offset dimensions don't
        match.
    """
    if connectivity is None:
        connectivity = 1

    if np.isscalar(connectivity):
        c_connectivity = ndi.generate_binary_structure(image_dim, connectivity)
    else:
        c_connectivity = np.array(connectivity, bool)
        if c_connectivity.ndim != image_dim:
            raise ValueError("Connectivity dimension must be same as image")

    if offset is None:
        if any([x % 2 == 0 for x in c_connectivity.shape]):
            raise ValueError("Connectivity array must have an unambiguous "
                             "center")

        offset = np.array(c_connectivity.shape) // 2

    return c_connectivity, offset


def _offsets_to_raveled_neighbors(image_shape, selem, center, order='C'):
    """Compute offsets to a samples neighbors if the image would be raveled.

    Parameters
    ----------
    image_shape : sequence
        The shape of the image for which the offsets are computed.
    selem : ndarray
        A structuring element determining the neighborhood expressed as an
        n-D array of 1's and 0's.
    center : sequence
        Indices specifying the center of `selem`.
    order : {"C", "F"}, optional
        Whether the image described by `image_shape` is in row-major (C-style)
        or column-major (Fortran-style) order.

    Returns
    -------
    raveled_offsets : ndarray
        Linear offsets to a samples neighbors in the raveled image, sorted by
        their distance from the center.

    Notes
    -----
    This function will return values even if `image_shape` contains a dimension
    length that is smaller than `selem`.

    Examples
    --------
    >>> _offsets_to_raveled_neighbors((4, 5), np.ones((4, 3)), (1, 1))
    array([-5, -1,  1,  5, -6, -4,  4,  6, 10,  9, 11])
    """
    selem_indices = np.array(np.nonzero(selem)).T
    offsets = selem_indices - center

    if order == 'F':
        offsets = offsets[:, ::-1]
        image_shape = image_shape[::-1]
    elif not order == 'C':
        raise ValueError("order was not 'C' or 'F'")

    # Scale offsets in each dimension and sum
    ravel_factors = image_shape[1:] + (1,)
    ravel_factors = np.cumprod(ravel_factors[::-1])[::-1]
    raveled_offsets = (offsets * ravel_factors).sum(axis=1)

    # Sort by distance
    distances = np.abs(offsets).sum(axis=1)
    raveled_offsets = raveled_offsets[np.argsort(distances)]

    # In case any dimension in image_shape is smaller than selem.shape
    # duplicates might occur, remove them
    if any(x < y for x, y in zip(image_shape, selem.shape)):
        # np.unique reorders, which we don't want
        _, indices = np.unique(raveled_offsets, return_index=True)
        raveled_offsets = raveled_offsets[np.sort(indices)]

    # Remove "offset to center"
    raveled_offsets = raveled_offsets[1:]

    return raveled_offsets


def _resolve_neighborhood(selem, connectivity, ndim):
    """Validate or create structuring element.

    Depending on the values of `connectivity` and `selem` this function
    either creates a new structuring element (`selem` is None) using
    `connectivity` or validates the given structuring element (`selem` is not
    None).

    Parameters
    ----------
    selem : array-like or None
        The structuring element to validate. See same argument in
        `local_maxima`.
    connectivity : int or None
        A number used to determine the neighborhood of each evaluated pixel.
        Defaults to `ndim` if `None` is given.
    ndim : int
        Number of dimensions `selem` ought to have.

    Returns
    -------
    selem : ndarray
        Validated or new structuring element specifying the neighborhood.
    """
    if selem is None:
        if connectivity is None:
            connectivity = ndim
        selem = ndi.generate_binary_structure(ndim, connectivity)
    else:
        # Validate custom structured element
        selem = np.asarray(selem, dtype=np.bool)
        # Must specify neighbors for all dimensions
        if selem.ndim != ndim:
            raise ValueError(
                "structuring element and image must have the same number of "
                "dimensions"
            )
        # Must only specify direct neighbors
        if any(s != 3 for s in selem.shape):
            raise ValueError("dimension size in structuring element is not 3")

    return selem


def _set_edge_values_inplace(image, value):
    """Set edge values along all axes to a constant value.

    Parameters
    ----------
    image : ndarray
        The array to modify inplace.
    value : scalar
        The value to use. Should be compatible with `image`'s dtype.

    Examples
    --------
    >>> image = np.zeros((4, 5), dtype=int)
    >>> _set_edge_values_inplace(image, 1)
    >>> image
    array([[1, 1, 1, 1, 1],
           [1, 0, 0, 0, 1],
           [1, 0, 0, 0, 1],
           [1, 1, 1, 1, 1]])
    """
    for axis in range(image.ndim):
        # Index first and last element in each dimension
        sl = (slice(None),) * axis + ((0, -1),) + (...,)
        image[sl] = value


def _fast_pad(image, value):
    """Pad an array on all axes by one with a value.

    Parameters
    ----------
    image : ndarray
        Image to pad.
    value : scalar
         The value to use. Should be compatible with `image`'s dtype.

    Returns
    -------
    padded_image : ndarray
        The new image.

    Notes
    -----
    The output of this function is equivalent to::

        np.pad(image, 1, mode="constant", constant_values=value)

    Up to version 1.17 `numpy.pad` uses concatenation to create padded arrays
    while this method needs to only allocate and copy once. This can result
    in significant speed gains if `image` has a large number of dimensions.
    Thus this function may be safely removed once that version is the minimum
    required by scikit-image.

    Examples
    --------
    >>> _fast_pad(np.zeros((2, 3), dtype=int), 4)
    array([[4, 4, 4, 4, 4],
           [4, 0, 0, 0, 4],
           [4, 0, 0, 0, 4],
           [4, 4, 4, 4, 4]])
    """
    # Allocate padded image
    new_shape = np.array(image.shape) + 2
    new_image = np.empty(new_shape, dtype=image.dtype, order="C")

    # Copy old image into new space
    sl = (slice(1, -1),) * image.ndim
    new_image[sl] = image
    # and set the edge values
    _set_edge_values_inplace(new_image, value)

    return new_image
