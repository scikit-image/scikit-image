"""
============
Thresholding
============

Thresholding is used to create a binary image.

This example uses Otsu's method to calculate the threshold value. Otsu's method
calculates an "optimal" threshold (marked by a red line in the histogram below)
by maximizing the variance between two classes of pixels, which are separated by
the threshold. Equivalently, this threshold minimizes the intra-class variance.

Additionally an adaptive thresholding is applied. Also known as local or
dynamic thresholding where the threshold value is the weighted mean for the
local neighborhood of a pixel subtracted by a constant. Small filter block sizes
are suitable for thresholding edges, large filter block sizes suitable for
thresholding larger homogeneous regions.

.. [1] http://en.wikipedia.org/wiki/Otsu's_method

"""

import matplotlib.pyplot as plt
import numpy as np

from skimage.data import camera
from skimage.filter import threshold_otsu, threshold_adaptive


image = camera()


#: Otsu thresholding
thresh = threshold_otsu(image)
otsu_binary = image > thresh

plt.figure(figsize=(8, 6))
plt.subplot(2, 3, 1)
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Original')
plt.axis('off')

plt.subplot(2, 3, 2, aspect='equal')
plt.hist(image)
plt.title('Histogram')
plt.axvline(thresh, color='r')

plt.subplot(2, 3, 3)
plt.imshow(otsu_binary, cmap=plt.cm.gray)
plt.title('Thresholded with Otsu')
plt.axis('off')


#: Adaptive thresholding
plt.subplot(2, 3, 4)
plt.imshow(threshold_adaptive(image, 11, 5, 'gaussian'), cmap=plt.cm.gray)
plt.title('Adaptive edge thresholding')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(threshold_adaptive(image, 125, 7.5, 'gaussian'), cmap=plt.cm.gray)
plt.title('Adaptive Gaussian')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(threshold_adaptive(image, 125, 7.5, 'mean'), cmap=plt.cm.gray)
plt.title('Adaptive Mean')
plt.axis('off')

plt.show()
