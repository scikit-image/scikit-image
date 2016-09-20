"""
=======================
Multi-Otsu Thresholding
=======================

Multi-Otsu threshold is used to separate the pixels of an input image
into three different classes, each one obtained according to the
intensity of the gray levels within the image.

Multi-Otsu calculates several thresholds, determined by the number of desired
classes. The default number of classes is 3: for obtaining three classes, the
algorithm returns two threshold values. They are represented by a red line in
the histogram below.

.. [1] Liao, P-S. and Chung, P-C., "A fast algorithm for multilevel
thresholding", Journal of Information Science and Engineering 17 (5):
713-727, 2001.
"""

from skimage import data
from skimage.filters import threshold_multiotsu

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Stablishing the font size for all plots.
matplotlib.rcParams['font.size'] = 9

# The input image.
image = data.camera()

# Applying multi-Otsu threshold for the default value, generating
# three classes.
thresh = threshold_multiotsu(image)

# Using the values on thresh, we generate the three regions.
region1 = image <= thresh[0]
region2 = (image > thresh[0]) & (image <= thresh[1])
region3 = image > thresh[1]

# Plotting the original image.
plt.figure(figsize=(8, 2.5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.axis('off')

# Plotting the histogram and the two thresholds obtained from
# multi-Otsu.
plt.subplot(1, 2, 2)
plt.hist(image)
plt.title('Histogram')
for i in range(len(thresh)):
    plt.axvline(thresh[i], color='r')

# Plotting the three resulting regions.
plt.figure(figsize=(9, 2.5))

plt.subplot(1, 3, 1)
plt.imshow(region1, cmap='gray')
plt.title('Multi-Otsu result, Region #1')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(region2, cmap='gray')
plt.title('Multi-Otsu result, Region #2')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(region3, cmap='gray')
plt.title('Multi-Otsu result, Region #3')
plt.axis('off')

plt.show()
