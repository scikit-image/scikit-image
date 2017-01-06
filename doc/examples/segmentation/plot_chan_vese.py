"""
======================
Chan-Vese Segmentation
======================

The Chan-Vese Algorithm is designed to segment objects without clearly
defined boundaries. This algorithm is based on level sets that are
evolved iteratively to minimize an energy, which is defined by
weighted values corresponding to the sum of differences inintensity
from the average value outside the segmented region, the sum of
differences from the average value inside the segmented region, and a
term which is dependent on the length of the boundary of the segmented
region.

This algorithm was first proposed by Tony Chan and Luminita Vese, in
a publicaion entitled "An Active Countour Model Without Edges" [1]_.

This implementation of the algorithm is somewhat simplified in the
sense that the area factor 'nu' described in the original paper is not
implemented, and is only suitable for grayscale images.

Typical values for lambda1 and lambda2 are 1. If the 'background' is
very different from the segmented object in terms of distribution (for
example, a uniform black image with figures of varying intensity), then
these values should be different from each other.

Typical values for mu are between 0 and 1, though higher values can be
used when dealing with shapes with very ill-defined contours.

The algorithm also returns a list of values which corresponds to the
energy at each iteration. This can be used to adjust the various
parameters described above.

References
----------
.. [1] An Active Contour Model without Edges, Tony Chan and
       Luminita Vese, Scale-Space Theories in Computer Vision, 1999
.. [2] Chan-Vese Segmentation, Pascal Getreuer Image Processing On
       Line, 2 (2012), pp. 214-224.
.. [3] The Chan-Vese Algorithm - Project Report, Rami Cohen
       http://arxiv.org/abs/1107.2782 , 2011
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import camera
from skimage.segmentation import chan_vese

image = camera().astype(np.float)
# Feel free to play around with these parameters to see how they impact the
# result
cv = chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, maxiter=1000,
               dt=0.5, starting_level_set="checkerboard", extended_output=True)
fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax[0, 0].imshow(image, cmap="gray", interpolation="nearest")
ax[0, 0].set_axis_off()
ax[0, 0].set_title("Original Image", fontsize=12)

ax[0, 1].imshow(cv[0], cmap="gray", interpolation="nearest")
ax[0, 1].set_axis_off()
ax[0, 1].set_title("Chan-Vese segmentation - "+str(len(cv[2]))+" iterations",
                   fontsize=12)

ax[1, 0].imshow(cv[1], cmap="gray", interpolation="nearest")
ax[1, 0].set_axis_off()
ax[1, 0].set_title("Final Level Set", fontsize=12)

ax[1, 1].plot(cv[2])
ax[1, 1].set_title("Evolution of energy for each iteration", fontsize=12)
fig.tight_layout()
plt.show()
