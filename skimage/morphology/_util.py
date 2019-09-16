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


def _offsets_to_raveled_neighbors(image_shape, structure, center, order='C'):
    """Compute offsets to a samples neighbors if the image would be raveled.

    Parameters
    ----------
    image_shape : tuple
        The shape of the image for which the offsets are computed.
    structure : ndarray
        A structuring element determining the neighborhood expressed as an
        n-D array of 1's and 0's.
    center : sequence
        Tuple of indices specifying the center of `selem`.

    Returns
    -------
    offsets : ndarray
        Linear offsets to a samples neighbors in the raveled image, sorted by
        their Euclidean distance from the center.

    Examples
    --------
    >>> _offsets_to_raveled_neighbors((4, 5), np.ones((4, 3)), (1, 1))
    array([-5, -1,  1,  5, -6, -4,  4,  6, 10,  9, 11])
    """
    structure = structure.copy()  # Don't modify original input
    structure[tuple(center)] = 0  # Ignore the center; it's not a neighbor
    connection_indices = np.transpose(np.nonzero(structure))
    offsets = (np.ravel_multi_index(connection_indices.T, image_shape,
                                    order=order) -
               np.ravel_multi_index(center, image_shape, order=order))
    squared_distances = np.sum((connection_indices - center) ** 2, axis=1)
    return offsets[np.argsort(squared_distances)]


def _resolve_neighborhood(selem, connectivity, ndim):
    """Validate or create structuring element for use in `local_maxima`.

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
        See same argument in `local_maxima`.
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
        sl = [slice(None)] * image.ndim
        # Set edge in front
        sl[axis] = 0
        image[tuple(sl)] = value
        # Set edge to the end
        sl[axis] = -1
        image[tuple(sl)] = value


def _fast_pad(image, value):
    """Pad an array on all axes with one constant value.

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

        np.pad(image, mode="constant", constant_values=value)

    However, this method needs to only allocate and copy once which can result
    in significant speed gains if `image` is large.

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
    original_slice = tuple(slice(1, -1) for _ in range(image.ndim))
    new_image[original_slice] = image
    # and set the edge values
    _set_edge_values_inplace(new_image, value)

    return new_image
