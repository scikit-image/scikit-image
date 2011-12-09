"""
============
Thresholding
============

Thresholding is used to create a binary image. This example uses Otsu's method to calculate the threshold value.

"""

import matplotlib.pyplot as plt

from skimage.data import camera
from skimage.thresholding import threshold_otsu


image = camera()
thresh = threshold_otsu(camera())
binary = image > thresh

plt.imshow(binary)
plt.axis('off')
plt.show()

