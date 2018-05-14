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
.. [2] Berger, C., Geraud, T., Levillain, R., Widynski, N., Baillard, A.,
       Bertin, E. (2007). Effective Component Tree Computation with
       Application to Pattern Recognition in Astronomical Imaging.
       In International Conference on Image Processing (ICIP) (pp. 41-44).
.. [3] Najman, L., & Couprie, M. (2006). Building the component tree in
       quasi-linear time. IEEE Transactions on Image Processing, 15(11),
       3531-3539.
.. [4] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
       Component Tree Computation Algorithms. IEEE Transactions on Image
       Processing, 23(9), 3885-3895.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skimage.morphology import build_max_tree
import numpy.random as rd
import pdb

def plot_img(image, ax, title, plot_text,
             image_values):
    ax.imshow(image, cmap='gray', aspect='equal', vmin=0, vmax=np.max(image))
    ax.axis('off')
    ax.set_title(title)

    for x in np.arange(-0.5, image.shape[0], 1.0):
        ax.add_artist(Line2D((x, x), (-0.5, image.shape[0]-0.5), color='blue', linewidth=2))

    for y in np.arange(-0.5, image.shape[1], 1.0):
        ax.add_artist(Line2D((-0.5, image.shape[1]), (y, y), color='blue', linewidth=2))
    
    if plot_text:
        k = 0
        for i in range(image_values.shape[0]):
            for j in range(image_values.shape[1]):
                ax.text(j, i, image_values[i,j], fontsize=8,
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='red')
                k += 1
    return

# small example image
image = np.array([[5, 4, 4, 3, 3], 
                  [5, 8, 8, 0, 0],
                  [2, 1, 2, 2, 1],
                  [4, 4, 4, 7, 5],
                  [5, 2, 3, 6, 6]],
                  dtype=np.uint8)
width, height = image.shape 

# raveled indices of the example image
raveled_indices = np.arange(np.prod(image.shape))
raveled_indices = raveled_indices.reshape(image.shape).astype(np.int)

# building the max-tree
P, S = build_max_tree(image)

# plot (example image and all possible thresholds)
fig, ax = plt.subplots(3, 9, figsize=(16, 6))

# top row: image (left: values printed on top, 
# right: raveled indices printed on top)
plot_img(image, ax[0, 0], 'Image Values',
         plot_text=True, image_values=image)
plot_img(image, ax[0,1], 'Raveled Indices',
         plot_text=True, 
         image_values=raveled_indices)
plot_img(P, ax[0,2], 'Max-tree indices',
         plot_text=True, 
         image_values=P)
plot_img(image, ax[0,3], 'Max-tree',
         plot_text=False, 
         image_values=raveled_indices)

# arrows corresponding to the max-tree
eps = 0.5
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        target_index = P[i,j]
        y = target_index // width
        x = target_index % width

        dx = x - j
        dy = y - i
        r = np.sqrt(dx**2 + dy**2)
        if r==0:
            # root of the tree
            continue
        dx = (r - eps) / r * dx
        dy = (r - eps) / r * dy
        delta = (i+j)%3
        ax[0,3].arrow(j + delta / 10.0, 
                      i + delta / 10.0, 
                      dx, 
                      dy, 
                      color='red', zorder=2,
                      head_width=0.2)

        if image[i,j] != image[y,x]:
            # in this case it is a canonical pixel.
            circle = plt.Circle((i, j), .4, color='r', fill=False)
            ax[0,3].add_artist(circle)
            
for k in range(3,9):
    ax[0,k].axis('off')
    
# application of all possible thresholds
for k in range(9):
    threshold = 8 - k
    bin_img = image >= threshold
    plot_img(bin_img, ax[1,k], 'Threhold : %i' % threshold,
             plot_text=True, image_values=raveled_indices)

    new_pixel_image = np.zeros(image.shape)
    new_pixel_image[image > threshold] = 255
    new_pixel_image[image == threshold] = 128
    plot_img(new_pixel_image, ax[2,k], '',
             plot_text=True, image_values=raveled_indices)

plt.tight_layout()
plt.show()

# In the second row, we see the results of a series of threshold operations
# The results are binary images with one or several connected components.
# The component tree is a tree representation of the connected components
# for the different thresholds. 
# A connected component A at threshold t is the parent of a connected
# component B at threshold t-1 if B is a subset of A.
# Here we have for instance: 
# {18, 23, 24} -> {18}, meaning that {18, 23, 24} is the parent of {18}
# {0, 5, 6, 7} -> {6, 7}
# The resulting tree is called a component tree.

# In the third row, we see a compacter representation, where we highlight for
# every threshold those pixels that have been added to the segmentation result,
# i.e. that have exactly the value of threshold.
# Here we would write:
# {23, 24} -> {18}
# {0, 5} -> {6, 7}
# This compact representation is called a max-tree.



