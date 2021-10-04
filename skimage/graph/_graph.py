import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse import csgraph
from ..morphology._util import _offsets_to_raveled_neighbors
from ..util._map_array import map_array


def pixel_graph(image=None, *, mask=None, edge_function=None):
    """Create an adjacency graph of pixels in an image.

    Parameters
    ----------
    image : array
        The input image. If the image is of type bool, it will be used as the
        mask as well.
    mask : array of bool
        Which pixels to use. If None, the graph for the whole image is used.
        If image is None and mask is not None, the mask is used as the image.
    edge_function : callable
        A function taking an array of pixel values, and an array of neighbor
        pixel values, and returning a value for the edge.

    Returns
    -------
    graph : scipy.sparse.csr_matrix
        A sparse adjacency matrix in which entry (i, j) is 1 if nodes i and j
        are neighbors, 0 otherwise.
    nodes : array of int
        The nodes of the graph. These correspond to the raveled indices of the
        nonzero pixels in the mask.
    """
    if mask is not None and image is None:
        image = mask
    if image.dtype == bool and mask is None:
        mask = image
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
        edge_function = np.subtract

    footprint = ndi.generate_binary_structure(mask.ndim, 1)
    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    nodes_padded = np.arange(padded.size).reshape(padded.shape)[padded]
    neighbor_offsets_padded = _offsets_to_raveled_neighbors(
            padded.shape, footprint, (1,) * padded.ndim
            )
    neighbors_padded = nodes_padded[:, np.newaxis] + neighbor_offsets_padded
    nodes = np.arange(mask.size).reshape(mask.shape)[mask]
    nodes_sequential = np.arange(nodes.size)
    # neighbors outside the mask get mapped to 0, which is a valid index,
    # BUT, they will be masked out in the next step.
    neighbors = map_array(neighbors_padded, nodes_padded, nodes)
    neighbors_mask = mask.ravel()[neighbors]
    num_neighbors = np.sum(neighbors_mask, axis=1)
    indices = np.repeat(nodes, num_neighbors)
    indices_sequential = np.repeat(nodes_sequential, num_neighbors)
    neighbor_indices = neighbors[neighbors_mask]
    neighbor_indices_sequential = map_array(
            neighbor_indices, nodes, nodes_sequential
            )
    if edge_function is None:
        data = np.broadcast_to(1, indices_sequential.shape)
    else:
        image_r = image.ravel()
        data = edge_function(image_r[indices], image_r[neighbor_indices])
    m = nodes_sequential.size
    mat = sparse.coo_matrix(
            (data, (indices_sequential, neighbor_indices_sequential)),
            shape=(m, m)
            )
    graph = mat.tocsr()
    return graph, nodes


def central_pixel(graph, nodes=None):
    if nodes is None:
        nodes = np.arange(graph.shape[0])
    all_shortest_paths = csgraph.shortest_path(graph, directed=False)
    all_shortest_paths_no_inf = np.nan_to_num(all_shortest_paths, posinf=0)
    total_shortest_path_len = np.sum(all_shortest_paths_no_inf, axis=1)
    nonzero = np.flatnonzero(total_shortest_path_len)
    min_sp = np.argmin(total_shortest_path_len[nonzero])
    return nodes[nonzero[min_sp]], total_shortest_path_len
