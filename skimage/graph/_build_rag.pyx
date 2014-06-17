import numpy as np
cimport numpy as cnp
import rag


def construct_rag_meancolor_3d(img, arr):
    """Computes the Region Adjacency Graph of a 3D color image using
    difference in mean color of regions as edge weights.

    Given an image and its segmentation, this method constructs the
    corresponsing Region Adjacency Graph (RAG). Each node in the RAG
    represents contiguous pixels with in `img` with the same label in
    `arr`. There is an edge between each pair of adjacent regions.

    Parameters
    ----------
    img : (width, height, depth, 3) ndarray
        Input image.
    arr : (width, height, depth) ndarray
        The array with labels.

    Returns
    -------
    out : RAG
        The region adjacency graph.
    """

    cdef Py_ssize_t depth,width,height, i, j, k
    cdef cnp.int32_t current, next
    width = arr.shape[0]
    height = arr.shape[1]
    depth = arr.shape[2]

    g = rag.RAG()

    i = 0
    for i in range(width-1):
        j = 0
        for j in range(height-1):
            k = 0
            for k in range(depth-1):
                current = arr[i, j, k]

                try:
                    g.node[current]['pixel count'] += 1
                    g.node[current]['total color'] += img[i, j]
                except KeyError:
                    g.add_node(current)
                    g.node[current]['pixel count'] = 1
                    g.node[current]['total color'] = img[i, j].astype(np.long)
                    g.node[current]['labels'] = [arr[i, j]]

                next = arr[i + 1, j, k]
                if current != next:
                    g.add_edge(current, next)

                next = arr[i, j + 1, k]
                if current != next:
                    g.add_edge(current, next)

                next = arr[i + 1, j + 1, k]
                if current != next:
                    g.add_edge(current, next)

                next = arr[i + 1, j, k + 1]
                if current != next:
                    g.add_edge(current, next)

                next = arr[i, j + 1, k + 1]
                if current != next:
                    g.add_edge(current, next)

                next = arr[i + 1, j + 1, k + 1]
                if current != next:
                    g.add_edge(current, next)

                next = arr[i, j, k + 1]
                if current != next:
                    g.add_edge(current, next)

                k += 1

            j += 1

        i += 1

    for n in g.nodes():
        g.node[n]['mean color'] = g.node[n][
            'total color'] / g.node[n]['pixel count']

    for x, y in g.edges_iter():
        diff = g.node[x]['mean color'] - g.node[y]['mean color']
        g[x][y]['weight'] = np.linalg.norm(diff)

    return g


def construct_rag_meancolor_2d(img, arr):
    """Computes the Region Adjacency Graph of a 2D color image using
    difference in mean color of regions as edge weights.

    Given an image and its segmentation, this method constructs the
    corresponsing Region Adjacency Graph (RAG). Each node in the RAG
    represents contiguous pixels with in `img` with the same label in
    `arr`. There is an edge between each pair of adjacent regions.

    Parameters
    ----------
    img : (width, height, 3) ndarray
        Input image.
    arr : (width, height) ndarray
        The array with labels.

    Returns
    -------
    out : RAG
        The region adjacency graph.
    """

    cdef Py_ssize_t width, height, h, i, j, k
    cdef cnp.int32_t current, next
    width = arr.shape[0]
    height = arr.shape[1]

    g = rag.RAG()

    i = 0
    for i in range(width-1):
        j = 0
        for j in range(height-1):
            current = arr[i, j]

            try:
                g.node[current]['pixel count'] += 1
                g.node[current]['total color'] += img[i, j]
            except KeyError:
                g.add_node(current)
                g.node[current]['pixel count'] = 1
                g.node[current]['total color'] = img[i, j].astype(np.long)
                g.node[current]['labels'] = [arr[i, j]]

            next = arr[i + 1, j]
            if current != next:
                g.add_edge(current, next)

            next = arr[i, j + 1]
            if current != next:
                g.add_edge(current, next)

            next = arr[i + 1, j + 1]
            if current != next:
                g.add_edge(current, next)

            j += 1

        i += 1

    for n in g.nodes():
        g.node[n]['mean color'] = g.node[n][
            'total color'] / g.node[n]['pixel count']

    for x, y in g.edges_iter():
        diff = g.node[x]['mean color'] - g.node[y]['mean color']
        g[x][y]['weight'] = np.linalg.norm(diff)

    return g
