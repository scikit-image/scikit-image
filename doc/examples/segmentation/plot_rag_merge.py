"""
===========
RAG Merging
===========

This example constructs a Region Adjacency Graph (RAG) and progressively merges
regions that are similar in color. Merging two adjacent regions produces
a new region with all the pixels from the merged regions. Regions are merged
until no highly similar region pairs remain.

"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, segmentation, color, util
from skimage.segmentation import graph


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])


image = util.img_as_float(data.eagle())
labels = segmentation.slic(
    image, compactness=0.5, n_segments=400, start_label=1, multichannel=False
    )
g = graph.rag_mean_color(image, labels)

labels2 = graph.merge_hierarchical(
    labels, g,
    thresh=3,
    rag_copy=True,
    in_place_merge=True,
    merge_func=merge_mean_color,
    weight_func=_weight_mean_color,
    )

out = color.label2rgb(labels2, image, kind='overlay', bg_label=0)
out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))

fig, ax = plt.subplots()
ax.imshow(out)
plt.show()
