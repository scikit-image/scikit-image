import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse import csgraph
from ..morphology._util import _offsets_to_raveled_neighbors
from ..util import map_array


def pixel_graph(mask):
    mask = mask.astype(bool)
    footprint = ndi.generate_binary_structure(mask.ndim, 1)
    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    nodes_padded = np.arange(padded.size).reshape(padded.shape)[padded]
    neighbor_offsets_padded = _offsets_to_raveled_neighbors(
            padded.shape, footprint, (1,) * padded.ndim
            )
    neighbors_padded = nodes_padded[:, np.newaxis] + neighbor_offsets_padded
    nodes = np.arange(mask.size).reshape(mask.shape)[mask]
    # neighbors outside the mask get mapped to 0, which is a valid index,
    # BUT, they will be masked out in the next step.
    neighbors = map_array(neighbors_padded, nodes_padded, nodes)
    neighbors_mask = mask.ravel()[neighbors]
    num_neighbors = np.sum(neighbors_mask, axis=1)
    indices = np.repeat(nodes, num_neighbors)
    neighbor_indices = neighbors[neighbors_mask]
    data = np.broadcast_to(1, indices.shape)
    m = mask.size
    mat = sparse.coo_matrix((data, (indices, neighbor_indices)), shape=(m, m))
    graph = mat.tocsr()
    return graph


def central_pixel(graph):
    all_shortest_paths = csgraph.shortest_path(graph, directed=False)
    all_shortest_paths_no_inf = np.nan_to_num(all_shortest_paths, posinf=0)
    total_shortest_path_len = np.sum(all_shortest_paths_no_inf, axis=1)
    nonzero = np.flatnonzero(total_shortest_path_len)
    min_sp = np.argmin(total_shortest_path_len[nonzero])
    return nonzero[min_sp], total_shortest_path_len
