"""
============
Thresholding
============

Thresholding is used to create a binary image. This example uses Otsu's method
to calculate the threshold value.

Otsu's method calculates an "optimal" threshold (marked by a red line in the
histogram below) by maximizing the variance between two classes of pixels,
which are separated by the threshold. Equivalently, this threshold minimizes
the intra-class variance.

.. [1] http://en.wikipedia.org/wiki/Otsu's_method

"""

import matplotlib.pyplot as plt

from skimage.data import camera
from skimage.filter import threshold_otsu


image = camera()
thresh = threshold_otsu(image)
binary = image > thresh

plt.figure(figsize=(8, 2.5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2, aspect='equal')
plt.hist(image)
plt.title('Histogram')
plt.axvline(thresh, color='r')

plt.subplot(1, 3, 3)
plt.imshow(binary, cmap=plt.cm.gray)
plt.title('Thresholded')
plt.axis('off')

plt.show()

