"""
===========
Shape Index
===========

The shape index is a single valued measure of local curvature,
derived from the eigen values of the Hessian, defined by Koenderink & van Doorn [1]_.

It can be used to find structures based on their apparent local shape.

Within this example, e.g. try to spot the guy-wires on the original image.
While only minutely visible in the input image, they become clearly recognizable
after calculating the shape index of the image.

.. [1] Koenderink, J. J. & van Doorn, A. J., "Surface shape and curvature scales",
       Image and Vision Computing, 1992, 10, 557-564. DOI:10.1016/0262-8856(92)90076-F
"""
import matplotlib.pyplot as plt

from skimage.data import rocket
from skimage.feature import shape_index


image = rocket().mean(axis=2)
s_1 = shape_index(image, sigma=1.0)
s_3 = shape_index(image, sigma=3.0)


fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(5, 15),
                                    sharex=True, sharey=True)

ax1.imshow(image, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('Input image', fontsize=18)

ax2.imshow(s_1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Shape index, $\sigma=1$', fontsize=18)

ax3.imshow(s_3, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Shape index, $\sigma=3$', fontsize=18)

fig.tight_layout()

plt.show()
