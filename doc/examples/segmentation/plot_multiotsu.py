"""
=======================
Multi-Otsu Thresholding
=======================

Multi-Otsu threshold [1]_ is used to separate the pixels of an input
image into three different classes, each one obtained according to the
intensity of the gray levels within the image.

Multi-Otsu calculates several thresholds, determined by the number of desired
classes. The default number of classes is 3: for obtaining three classes, the
algorithm returns two threshold values. They are represented by a red line in
the histogram below.

.. [1] Liao, P-S., Chen, T-S. and Chung, P-C., "A fast algorithm for multilevel
       thresholding", Journal of Information Science and Engineering 17 (5):
       713-727, 2001. Available at:
       <http://ftp.iis.sinica.edu.tw/JISE/2001/200109_01.pdf>.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage.filters import threshold_multiotsu

# Stablishing the font size for all plots.
matplotlib.rcParams['font.size'] = 9

# The input image.
image = data.camera()

# Applying multi-Otsu threshold for the default value, generating
# three classes.
thresh, _ = threshold_multiotsu(image)

# Using the values on thresh, we generate the three regions.
regions = np.digitize(image, bins=thresh)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 8))

# Plotting the original image.
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')

# Plotting the histogram and the two thresholds obtained from
# multi-Otsu.
ax[1].hist(image.ravel())
ax[1].set_title('Histogram')
for i in range(len(thresh)):
    ax[1].axvline(thresh[i], color='r')

# Plotting the Multi Otsu result.
ax[2].imshow(regions, cmap='Accent')
ax[2].set_title('Multi-Otsu result')
ax[2].axis('off')

plt.show()
