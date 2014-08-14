import numpy as np


def _hmerge(rag, x, y, n):
    diff = rag.node[y]['mean color'] - rag.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return diff


def merge_hierarchical(rag, labels, thresh):

    min_wt = 0
    while min_wt < thresh:

        valid_edges = ((x, y, d)
                       for x, y, d in rag.edges(data=True) if x != y)
        x, y, d = min(valid_edges, key=lambda x: x[2]['weight'])
        min_wt = d['weight']

        if min_wt < thresh:
            total_color = (rag.node[y]['total color'] +
                           rag.node[x]['total color'])
            n_pixels = rag.node[x]['pixel count'] + rag.node[y]['pixel count']
            rag.node[y]['total color'] = total_color
            rag.node[y]['pixel count'] = n_pixels
            rag.node[y]['mean color'] = total_color / n_pixels

            rag.merge_nodes(x, y, _hmerge)

    count = 0
    arr = np.arange(labels.max() + 1)
    for n, d in rag.nodes_iter(data=True):
        for l in d['labels']:
            arr[l] = count
        count += 1

    return arr[labels]
