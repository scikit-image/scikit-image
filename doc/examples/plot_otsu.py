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

plt.figure(figsize=(10, 3.5))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.hist(image)
plt.title('histogram')
plt.axvline(thresh, color='r')

plt.subplot(1, 3, 3)
plt.imshow(binary)
plt.title('thresholded')
plt.axis('off')

plt.show()

