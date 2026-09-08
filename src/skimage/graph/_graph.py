from _skimage2.graph._graph import (
    central_pixel as central_pixel,
)  # noqa: F401

import scipy.sparse as sparse

import _skimage2 as ski2

from .._migration import ski2_migration_decorator

__all__ = [
    'central_pixel',
    'pixel_graph',
]

from skimage._doctest_adapters import adapt_doctests  # noqa: E402

adapt_doctests(globals())


@ski2_migration_decorator(
    """\
    ``%(qname_old)s`` is deprecated in favor of
    ``%(qname_new)s`` with new behavior:

    * Parameter `sparse_type` was removed
    * Returned `graph` is always a `scipy.sparse.csr_array`

    SciPy 2.0 deprecates the sparse matrix classes and will remove them no
    earlier than SciPy 2.2. The two types define ``*`` differently: matrix
    multiplication for `scipy.sparse.csr_matrix`, elementwise multiplication
    for `scipy.sparse.csr_array`. ``@`` is matrix multiplication for both.

    To keep the old (``skimage``, v1.x) behavior, convert the result::

        graph, nodes = skimage2.graph.pixel_graph(image, ...)
        graph = scipy.sparse.csr_matrix(graph)
    """,
    qname_old="skimage.graph.pixel_graph",
)
def pixel_graph(
    image,
    *,
    mask=None,
    edge_function=None,
    connectivity=1,
    spacing=None,
    sparse_type="matrix",
):
    """Create an adjacency graph of pixels in an image.

    Pixels where the mask is True are nodes in the returned graph, and they are
    connected by edges to their neighbors according to the connectivity
    parameter. By default, the *value* of an edge when a mask is given, or when
    the image is itself the mask, is the Euclidean distance between the pixels.

    However, if an int- or float-valued image is given with no mask, the value
    of the edges is the absolute difference in intensity between adjacent
    pixels, weighted by the Euclidean distance.

    Parameters
    ----------
    image : array
        The input image. If the image is of type bool, it will be used as the
        mask as well.
    mask : array of bool
        Which pixels to use. If None, the graph for the whole image is used.
    edge_function : callable
        A function taking an array of pixel values, and an array of neighbor
        pixel values, and an array of distances, and returning a value for the
        edge. If no function is given, the value of an edge is just the
        distance.
    connectivity : int
        The square connectivity of the pixel neighborhood: the number of
        orthogonal steps allowed to consider a pixel a neighbor. See
        `scipy.ndimage.generate_binary_structure` for details.
    spacing : tuple of float
        The spacing between pixels along each axis.
    sparse_type : {"matrix", "array"}, optional
        The return type of `graph`, either `scipy.sparse.csr_array` or
        `scipy.sparse.csr_matrix` (default).

    Returns
    -------
    graph : scipy.sparse.csr_matrix or scipy.sparse.csr_array
        A sparse adjacency matrix in which entry (i, j) is 1 if nodes i and j
        are neighbors, 0 otherwise. Depending on `sparse_type`, this can be
        returned as a `scipy.sparse.csr_array`.
    nodes : array of int
        The nodes of the graph. These correspond to the raveled indices of the
        nonzero pixels in the mask.
    """
    if sparse_type not in ("array", "matrix"):
        msg = f"`sparse_type` must be 'array' or 'matrix', got {sparse_type}"
        raise ValueError(msg)

    graph, nodes = ski2.graph.pixel_graph(
        image,
        mask=mask,
        edge_function=edge_function,
        connectivity=connectivity,
        spacing=spacing,
    )
    if sparse_type == "matrix":
        graph = sparse.csr_matrix(graph)

    return graph, nodes
