"""
==============
Edge operators
==============

Edge operators are used in image processing within edge detection algorithms.
They are discrete differentiation operators, computing an approximation of the
gradient of the image intensity function.

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage.data import camera
from skimage.filters import roberts, sobel, scharr


image = camera()
edge_roberts = roberts(image)
edge_sobel = sobel(image)

fig, (ax0, ax1) = plt.subplots(ncols=2)

ax0.imshow(edge_roberts, cmap=plt.cm.gray)
ax0.set_title('Roberts Edge Detection')
ax0.axis('off')

ax1.imshow(edge_sobel, cmap=plt.cm.gray)
ax1.set_title('Sobel Edge Detection')
ax1.axis('off')

plt.tight_layout()

"""
.. image:: PLOT2RST.current_figure

Different operators compute different finite-difference approximations of the
gradient. For example, the Scharr filter results in a better rotational
variance than other filters such as the Sobel filter [1]_ [2]_. The difference
between the two filters is illustrated below on an image that is the
discretization of a rotation-invariant continuous function. The discrepancy
between the two filters is stronger for regions of the image where the
direction of the gradient is close to diagonal, and for regions with high
spatial frequencies.

.. [1] http://en.wikipedia.org/wiki/Sobel_operator#Alternative_operators

.. [2] B. Jaehne, H. Scharr, and S. Koerkel. Principles of filter design. In
       Handbook of Computer Vision and Applications. Academic Press, 1999.
"""

x, y = np.ogrid[:100, :100]
# Rotation-invariant image with different spatial frequencies
img = np.exp(1j * np.hypot(x, y)**1.3 / 20.).real

edge_sobel = sobel(img)
edge_scharr = scharr(img)

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)

ax0.imshow(edge_sobel, cmap=plt.cm.gray)
ax0.set_title('Sobel Edge Detection')
ax0.axis('off')

ax1.imshow(edge_scharr, cmap=plt.cm.gray)
ax1.set_title('Scharr Edge Detection')
ax1.axis('off')

ax2.imshow(img, cmap=plt.cm.gray)
ax2.set_title('Original image')
ax2.axis('off')

ax3.imshow(edge_scharr - edge_sobel, cmap=plt.cm.jet)
ax3.set_title('difference (Scharr - Sobel)')
ax3.axis('off')

plt.tight_layout()

plt.show()

"""
.. image:: PLOT2RST.current_figure
"""
