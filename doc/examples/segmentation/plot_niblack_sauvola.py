"""
================================
Niblack and Sauvola Thresholding
================================

Niblack and Sauvola thresholds are local thresholding techniques that are
useful for images where the background is not uniform, especially for text
recognition [1]_, [2]_. Instead of calculating a single global threshold for
the entire image, several thresholds are calculated for every pixel by using
specific formulae that take into account the mean and standard deviation of the
local neighborhood (defined by a window centered around the pixel).

Here, we binarize an image using these algorithms compare it to a common global
thresholding technique. Parameter `window_size` determines the size of the
window that contains the surrounding pixels.

.. [1] Niblack, W (1986), An introduction to Digital Image
       Processing, Prentice-Hall.
.. [2] J. Sauvola and M. Pietikainen, "Adaptive document image
       binarization," Pattern Recognition 33(2),
       pp. 225-236, 2000.
       :DOI:`10.1016/S0031-3203(99)00055-2`
"""

import matplotlib
import matplotlib.pyplot as plt

from skimage.data import page
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola


matplotlib.rcParams['font.size'] = 9


image = page()
binary_global = image > threshold_otsu(image)

window_size = 25
thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
thresh_sauvola = threshold_sauvola(image, window_size=window_size)

binary_niblack = image > thresh_niblack
binary_sauvola = image > thresh_sauvola

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))

axes[0, 0] = plt.subplot(2, 2, 1)
axes[0, 0].imshow(image, cmap=plt.cm.gray)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

axes[0, 1] = plt.subplot(2, 2, 2)
axes[0, 1].imshow(binary_global, cmap=plt.cm.gray)
axes[0, 1].set_title('Global Threshold')
axes[0, 1].axis('off')

axes[1, 0] = plt.subplot(2, 2, 3)
axes[1, 0].imshow(binary_niblack, cmap=plt.cm.gray)
axes[1, 0].set_title('Niblack Threshold')
axes[1, 0].axis('off')

axes[1, 1] = plt.subplot(2, 2, 4)
axes[1, 1].imshow(binary_sauvola, cmap=plt.cm.gray)
axes[1, 1].set_title('Sauvola Threshold')
axes[1, 1].axis('off')

plt.show()
