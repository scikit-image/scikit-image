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
