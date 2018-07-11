"""
========
Max-tree
========
The max-tree is a hierarchical representation of an image that is the basis
for a large family of morphological filters.

If we apply a threshold operation to an image, we obtain a binary image
containing a certain number of connected components. If we apply a lower
threshold, we observe that the connected components we obtain now contain
all connected components obtained by the higher threshold. With this, we can
define a graph representation of the components: whenever a connected component
A obtained by thresholding with threshold t1 is contained in a component B
obtained by thresholding with threshold t1 < t2, we say that B is the parent
of A. The resulting tree structure is called a component tree. The max-tree
is a compact representation of such a component tree.

In this example we give an intuition what a max-tree is.

References
----------
.. [1] Salembier, P., Oliveras, A., & Garrido, L. (1998). Antiextensive
       Connected Operators for Image and Sequence Processing.
       IEEE Transactions on Image Processing, 7(4), 555-570.
       DOI:10.1109/83.663500
.. [2] Berger, C., Geraud, T., Levillain, R., Widynski, N., Baillard, A.,
       Bertin, E. (2007). Effective Component Tree Computation with
       Application to Pattern Recognition in Astronomical Imaging.
       In International Conference on Image Processing (ICIP) (pp. 41-44).
       DOI:10.1109/ICIP.2007.4379949
.. [3] Najman, L., & Couprie, M. (2006). Building the component tree in
       quasi-linear time. IEEE Transactions on Image Processing, 15(11),
       3531-3539.
       DOI:10.1109/TIP.2006.877518
.. [4] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
       Component Tree Computation Algorithms. IEEE Transactions on Image
       Processing, 23(9), 3885-3895.
       DOI:10.1109/TIP.2014.2336551
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skimage.morphology import max_tree
import networkx as nx
from matplotlib.gridspec import GridSpec


# plot_img is a helper function to plot the images and overlay the image
# values or indices.
def plot_img(image, ax, title, plot_text,
             image_values):

    ax.imshow(image, cmap='gray', aspect='equal', vmin=0, vmax=np.max(image))
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xticks([])

    for x in np.arange(-0.5, image.shape[0], 1.0):
        ax.add_artist(Line2D((x, x), (-0.5, image.shape[0]-0.5),
                             color='blue', linewidth=2))

    for y in np.arange(-0.5, image.shape[1], 1.0):
        ax.add_artist(Line2D((-0.5, image.shape[1]), (y, y),
                             color='blue', linewidth=2))

    if plot_text:
        k = 0
        for i in range(image_values.shape[0]):
            for j in range(image_values.shape[1]):
                ax.text(j, i, image_values[i, j], fontsize=8,
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='red')
                k += 1
    return


# prune transforms a canonical max-tree to a max-tree.
def prune(G, node, res):
    value = G.nodes[node]['value']
    res[node] = str(node)
    preds = [p for p in G.predecessors(node)]
    for p in preds:
        if (G.nodes[p]['value'] == value):
            res[node] += ', %i' % p
            G.remove_node(p)
        else:
            prune(G, p, res)
    G.nodes[node]['label'] = res[node]
    return


# accumulate transforms a max-tree to a component tree.
def accumulate(G, node, res):
    total = G.nodes[node]['label']
    parents = G.predecessors(node)
    for p in parents:
        total += ', ' + accumulate(G, p, res)
    res[node] = total
    return total


# sets the positions of a max-tree. This is necessary, as we wanted to
# visually distinguish between nodes at the same level and nodes at
# different levels.
def position_nodes_for_max_tree(G, image_rav, root_x=4, delta_x=1.2):
    pos = {}
    for node in reversed(list(nx.topological_sort(CMT))):
        value = G.nodes[node]['value']
        if CMT.out_degree(node) == 0:
            # root
            pos[node] = (root_x, value)

        in_nodes = [y for y in CMT.predecessors(node)]

        # place the nodes at the same level
        level_nodes = [y for y in
                       filter(lambda x: image_rav[x] == value, in_nodes)]
        nb_level_nodes = len(level_nodes) + 1

        c = nb_level_nodes // 2
        i = - c
        if (len(level_nodes) < 3):
            hy = 0
            m = 0
        else:
            hy = 0.25
            m = hy / (c - 1)

        for level_node in level_nodes:
            if(i == 0):
                i += 1
            if (len(level_nodes) < 3):
                pos[level_node] = (pos[node][0] + i * 0.6 * delta_x, value)
            else:
                pos[level_node] = (pos[node][0] + i * 0.6 * delta_x,
                                   value + m * (2 * np.abs(i) - c - 1))
            i += 1

        # place the nodes at different levels
        other_level_nodes = [y for y in
                             filter(lambda x: image_rav[x] > value, in_nodes)]
        if (len(other_level_nodes) == 1):
            i = 0
        else:
            i = - len(other_level_nodes) // 2
        for other_level_node in other_level_nodes:
            if((len(other_level_nodes) % 2 == 0) and (i == 0)):
                i += 1
            pos[other_level_node] = (pos[node][0] + i * delta_x,
                                     image_rav[other_level_node])
            i += 1

    return pos

#####################################################################
# First, we define a small test image.
# For clarity, we choose an example image, where image values cannot be
# confounded with indices (different range).
image = np.array([[40, 40, 39, 39, 38],
                  [40, 41, 39, 39, 39],
                  [30, 30, 30, 32, 32],
                  [33, 33, 30, 32, 35],
                  [30, 30, 30, 33, 36]], dtype=np.uint8)

# the raveled image (for display purpose)
image_rav = image.ravel()

# raveled indices of the example image (for display purpose)
raveled_indices = np.arange(np.prod(image.shape))
raveled_indices = raveled_indices.reshape(image.shape).astype(np.int)

# max-tree of the image
P, S = max_tree(image)

P_rav = P.ravel()


# Now, we plot the image, all possible thresholds and the corresponding
# trees (component tree and max-tree).
fig = plt.figure(figsize=(18, 9))

gs = GridSpec(3, 9,
              left=0.05, right=0.95,
              top=0.95, bottom=0.05,
              height_ratios=[1.5, 1, 3])
gs.update(wspace=0.05, hspace=0.2)

#####################################################################
# First row of the plot
# Here we plot the image with the following overlays:
# - the image values
# - the raveled indices (serve as pixel identifiers)
# - the output of the max_tree function
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[0, 2])
ax4 = plt.subplot(gs[0, 3:])
ax4.axis('off')

plot_img(image - image.min(), ax1, 'Image Values',
         plot_text=True, image_values=image)
plot_img(image - image.min(), ax2, 'Raveled Indices',
         plot_text=True,
         image_values=raveled_indices)
plot_img(image - image.min(), ax3, 'Max-tree indices',
         plot_text=True,
         image_values=P)


#####################################################################
# Second row of the plot
# Here, we see the results of a series of threshold operations.
# The component tree (and max-tree) is a representation of the inclusion
# of the corresponding pixel sets.

thresholds = np.unique(image)
for k, threshold in enumerate(thresholds.tolist()):
    bin_img = image >= threshold
    ax = plt.subplot(gs[1, k])
    plot_img(bin_img, ax, 'Threhold : %i' % threshold,
             plot_text=True, image_values=raveled_indices)


#####################################################################
# Third row of the plot
# Here, we plot the component and max-trees. A component tree relates
# the different pixel sets resulting from all possible threshold operations
# to each other. There is an arrow in the graph, if a component at one level
# is included in the component of a lower level. The max-tree is just
# a different encoding of the pixel sets.
# 1. the component tree: pixel sets are explicitly written out. We see for
#    instance that {6} (result of applying a threshold at 41) is the parent
#    of {0, 1, 5, 6} (threshold at 40).
# 2. the max-tree: only pixels that come into the set at this level
#    are explicitly written out. We therefore will write
#    {6} -> {0,1,5} instead of {6} -> {0, 1, 5, 6}
# 3. the canonical max-treeL this is the representation which is given by
#    our implementation. Here, every pixel is a node. Connected components
#    are represented by one of the pixels. We thus replace
#    {6} -> {0,1,5} by {6} -> {5}, {1} -> {5}, {0} -> {5}
#    This allows us to represent the graph by an image (top row, third column).


##############################
# the canonical max-tree graph
CMT = nx.DiGraph()
CMT.add_nodes_from(S)
for node in CMT.nodes():
    CMT.nodes[node]['value'] = image_rav[node]
CMT.add_edges_from([(n, P_rav[n]) for n in S[1:]])

# max-tree from the canonical max-tree
MT = nx.DiGraph(CMT)
labels = {}
prune(MT, S[0], labels)

# component tree from the max-tree
labels_ct = {}
total = accumulate(MT, S[0], labels_ct)

# positions of nodes : canonical max-tree (CMT)
pos_cmt = position_nodes_for_max_tree(CMT, image_rav)

# positions of nodes : max-tree (MT)
pos_mt = dict(zip(MT.nodes, [pos_cmt[node] for node in MT.nodes]))

# plot the trees with networkx and matplotlib
ax1 = plt.subplot(gs[2, :3])
ax2 = plt.subplot(gs[2, 3:6], sharey=ax1)
ax3 = plt.subplot(gs[2, 6:], sharey=ax2)

# component tree
plt.sca(ax1)
nx.draw_networkx(MT, pos=pos_mt,
                 node_size=40, node_shape='s', node_color='white',
                 font_size=6, labels=labels_ct)
for v in range(image_rav.min(), image_rav.max() + 1):
    plt.axhline(v - 0.5, linestyle=':')
    plt.text(-3, v - 0.15, "value : %i" % v, fontsize=8)
plt.axhline(v + 0.5, linestyle=':')
plt.xlim(xmin=-3, xmax=10)
plt.title('Component tree')
plt.axis('off')

# max-tree
plt.sca(ax2)
nx.draw_networkx(MT, pos=pos_mt,
                 node_size=40, node_shape='s', node_color='white',
                 font_size=8, labels=labels)
for v in range(image_rav.min(), image_rav.max() + 1):
    plt.axhline(v - 0.5, linestyle=':')
    plt.text(0, v - 0.15, "value : %i" % v, fontsize=8)
plt.axhline(v + 0.5, linestyle=':')
plt.xlim(xmin=0)
plt.title('Max tree')
plt.axis('off')

# canonical max-tree
plt.sca(ax3)
nx.draw_networkx(CMT, pos=pos_cmt,
                 node_size=40, node_shape='s', node_color='white',
                 font_size=8)
for v in range(image_rav.min(), image_rav.max() + 1):
    plt.axhline(v - 0.5, linestyle=':')
    plt.text(0, v - 0.15, "value : %i" % v, fontsize=8)
plt.axhline(v + 0.5, linestyle=':')
plt.xlim(xmin=0)
plt.title('Canonical max tree')
plt.axis('off')

plt.show()
