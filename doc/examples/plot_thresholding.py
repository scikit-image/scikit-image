"""
============
Thresholding
============

Thresholding is used to create a binary image.

This example uses Otsu's method to calculate the threshold value. Otsu's method
calculates an "optimal" threshold (marked by a red line in the histogram below)
by maximizing the variance between two classes of pixels, which are separated by
the threshold. Equivalently, this threshold minimizes the intra-class variance.

Additionnally an adaptive thresholding is applied. Also known as local or
dynamic thresholding where the the threshold value is the weighted mean for the
local neighborhood of a pixel subtracted by a constant.

.. [1] http://en.wikipedia.org/wiki/Otsu's_method

"""

import matplotlib.pyplot as plt
import numpy as np

from skimage.data import camera
from skimage.filter import threshold_otsu, adaptive_threshold


image = camera()
thresh = threshold_otsu(image)
otsu_binary = image > thresh
adaptive_binary = np.invert(adaptive_threshold(image, 9, 5))

plt.figure(figsize=(8, 2.5))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Original')
plt.axis('off')

plt.subplot(2, 2, 2, aspect='equal')
plt.hist(image)
plt.title('Histogram')
plt.axvline(thresh, color='r')

plt.subplot(2, 2, 3)
plt.imshow(otsu_binary, cmap=plt.cm.gray)
plt.title('Thresholded with Otsu')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(adaptive_binary, cmap=plt.cm.gray)
plt.title('Adaptively thresholded')
plt.axis('off')

plt.show()


