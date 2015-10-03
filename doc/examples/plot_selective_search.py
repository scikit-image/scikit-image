"""
================
Selective Search
================

Selective search uses an heirachical groupings of an initial oversegmented image
to propose regions in an image that may contian objects.  By grouping adjacent
regions using a region-level similarity measure you can generate many
approximate locations essientally capturing objects at all scales.

This example implements a simplified version of Selective Search [1] by first
constructing a Region Adjacency Graph (RAG).  The RAG is progressively merges
regions that are similar in color. Merging two adjacent regions produces a new
region with all the pixels from the merged regions. Regions are merged until no
highly similar region pairs remain.  Unlike [1], this example does not further
processing, that is, we do not classify the regions.

This examaple is based RAG Merging [2] and uses the same callbacks. For more
powerful merging strategy see Selective Search [1] for equations that uses size,
texture and colour to influence the merging of adjacent regions.

References
----------

.. [1] Uijlings, Jasper RR, et al. "Selective search for object recognition."
       International journal of computer vision 104.2 (2013): 154-171.

.. [2] RAG Merging, scikit-image Gallery.

"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.future import graph
from skimage import measure, data, segmentation, color


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
    weight : float
        The absolute difference of the mean color between node `dst` and `n`.
    """
    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return diff


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
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])


def add_bbox(labels, boxes):
    """Find the unique bounding boxes in a labeled image

    Parameters
    ----------
    labels: ndarray
        Labeled image
    bbox: list
       current list of known bounding boxes.

    Returns
    -------
    If a new boundng box is found it is appended to `bbox`
    """
    for props in measure.regionprops(labels):
        if props.bbox not in boxes:
            boxes.append(props.bbox)


# Original paper uses multiple colour spaces with complementary invariance
# properties.  Fro this example we just use two, RGB and Intensity.
img = data.astronaut()
gray = color.rgb2gray(data.astronaut())
source = [img,gray]

detections = []
for image in source:
    labels = segmentation.slic(image, compactness=20, n_segments=200)
    g = graph.rag_mean_color(image, labels)
    merged, steps = graph.merge_hierarchical(labels, g, thresh=400,
                                             rag_copy=True,
                                             in_place_merge=False,
                                             weight_func=_weight_mean_color,
                                             merge_func=merge_mean_color,
                                             merge_trace=True)

    # Add regions from initial segmentation (smallest scale)
    add_bbox(labels, detections)

    # Inspect each step and add bounding box
    label_map = np.arange(labels.max() + 1)
    for step in steps:
        label_map[:] = 2
        for label in step:
            label_map[label] = 1

        add_bbox(label_map[labels], detections)

# Selective Search has a different goal from segmentation, the result is
# list of proposed regions that may contain an object, not a final segmentaiton.
# Here we plot the last 10 bounding boxes.  Each bounding box encloses a
# region found during the merge process.

fig, ax = plt.subplots()
plt.imshow(img)
for (minr, minc, maxr, maxc) in detections[-10:]:
    patch = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                               facecolor="none", edgecolor="blue", linewidth=2)
    ax.add_patch(patch)
plt.show()
