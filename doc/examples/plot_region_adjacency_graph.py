"""
=================================
RAG Construction and Thresholding
=================================

This example demonstrates how to constructs a Region Adjacency Graph (RAG) using
a user defined function to describe the nodes, edges and calculate the edge
weights.

We duplicate the functionality of the RAG Thresholding example from the skimage
gallery [1].  That is, We construct a RAG and in the user specified function
define edges as the difference in mean color. We then join regions with similar
mean color.

References
----------

.. [1] RAG Merging, scikit-image Gallery.

"""

import numpy as np
from skimage import data, io, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt


def define_mean_color_rag(graph, labels, image, extra_arguments=[],
                           extra_keywords={'sigma':255.0, 'mode':'distance'}):
        """Callback to handle describing nodes.

        Nodes can have arbitrary Python objects assigned as attributes. This
        method expects a valid graph and computes the mean color of the node.

        Parameters
        ----------
        graph : RAG
            The graph under consideration.
        labels : ndarray, shape(M, N, [..., P,])
            The labelled image. This should have one dimension less than
            `image`. If `image` has dimensions `(M, N, 3)` `labels` should have
            dimensions `(M, N)`.
        image : ndarray, shape(M, N, [..., P,] 3)
            Input image.
        extra_arguments : sequence, optional
            Allows extra positional arguments passed.
        extra_keywords : dictionary, optional
            Allows extra keyword arguments passed.

        """
        # Describe the nodes
        for n in graph:
            graph.node[n].update({'labels': [n],
                                'pixel count': 0,
                                'total color': np.array([0, 0, 0],
                                                        dtype=np.double)})

        for index in np.ndindex(labels.shape):
            current = labels[index]
            graph.node[current]['pixel count'] += 1
            graph.node[current]['total color'] += image[index]

        for n in graph:
            graph.node[n]['mean color'] = (graph.node[n]['total color'] /
                                        graph.node[n]['pixel count'])

        # Calcuate the edge weights
        sigma = extra_keywords['sigma']
        mode = extra_keywords['mode']

        for x, y, d in graph.edges_iter(data=True):
            diff = graph.node[x]['mean color'] - graph.node[y]['mean color']
            diff = np.linalg.norm(diff)
            if mode == 'similarity':
                d['weight'] = math.e ** (-(diff ** 2) / sigma)
            elif mode == 'distance':
                d['weight'] = diff
            else:
                raise ValueError("The mode '%s' is not recognised" % mode)



img = data.coffee()

labels1 = segmentation.slic(img, compactness=30, n_segments=400)
out1 = color.label2rgb(labels1, img, kind='avg')

extra_keywords={'sigma':255.0, 'mode':'distance'}
rag = graph.region_adjacency_graph(labels1, image=img, connectivity=2,
                               describe_func=define_mean_color_rag,
                               extra_arguments=[],
                               extra_keywords=extra_keywords)
labels2 = graph.cut_threshold(labels1, rag, 29)
out2 = color.label2rgb(labels2, img, kind='avg')

plt.figure()
io.imshow(out1)
plt.figure()
io.imshow(out2)
io.show()
