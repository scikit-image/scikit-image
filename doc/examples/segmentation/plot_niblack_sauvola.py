"""
============================================
Niblack, Sauvola and Phansalkar Thresholding
============================================

Niblack, Sauvola and Phansalkar thresholds are local thresholding techniques
that are useful for images where the background is not uniform. Niblack and
Sauvola thresholds are especially usefull for text recognition [1]_, [2]_,
while the Phansalkar threshold was originally designed for detection of cell
nuclei in low contrast images [3]_. Instead of calculating a single global
threshold for the entire image, several thresholds are calculated for every
pixel by using specific formulae that take into account the mean and standard
deviation of the local neighborhood (defined by a window centered around the
pixel).

Here, we binarize an image using these algorithms compare it to a common global
thresholding technique. Parameter `window_size` determines the size of the
window that contains the surrounding pixels.

.. [1] Niblack, W (1986), An introduction to Digital Image
       Processing, Prentice-Hall.
.. [2] J. Sauvola and M. Pietikainen, "Adaptive document image
       binarization," Pattern Recognition 33(2),
       pp. 225-236, 2000.
       :DOI:`10.1016/S0031-3203(99)00055-2`
.. [3] Phansalskar N. et al. "Adaptive local thresholding for detection of
       nuclei in diversity stained cytology images.", International Conference
       on Communications and Signal Processing (ICCSP),
       pp. 218-220, 2011
       :DOI:`10.1109/ICCSP.2011.5739305`
"""
import matplotlib
import matplotlib.pyplot as plt

from skimage.data import page, moon
from skimage.exposure import equalize_adapthist
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola, threshold_phansalkar)


matplotlib.rcParams['font.size'] = 9

# use image = moon() to see the advantage of Phansalkar threshold
image = page()
binary_global = image > threshold_otsu(image)

window_size = 25
thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
thresh_sauvola = threshold_sauvola(image, window_size=window_size)

# The Phansalkar threshold requires a an image with an equalized histogram.
# Contrast Limited Adaptive Histogram Equalization shows the best results.
image_eq = equalize_adapthist(image)

# The Phansalkar threshold can additionally be modified using the parameters
# p and q. According to the authors, the parameter p can be changed
# from 1.0 to 5.0, while 3.0 gave the best results for their usecase. 
thresh_phansalkar_p1 = threshold_phansalkar(image_eq, window_size=window_size, p=1.5)
thresh_phansalkar_p3 = threshold_phansalkar(image_eq, window_size=window_size, p=3.0)

#process binary masks
binary_niblack = image > thresh_niblack
binary_sauvola = image > thresh_sauvola
binary_phansalkar_p1 = image_eq > thresh_phansalkar_p1
binary_phansalkar_p3 = image_eq > thresh_phansalkar_p3

plt.figure(figsize=(10, 7))
plt.subplot(3, 2, 1)
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Original')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.title('Global Threshold')
plt.imshow(binary_global, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(3, 2, 3)
plt.imshow(binary_niblack, cmap=plt.cm.gray)
plt.title('Niblack Threshold')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(binary_sauvola, cmap=plt.cm.gray)
plt.title('Sauvola Threshold')
plt.axis('off')

plt.subplot(3, 2, 5)
plt.imshow(binary_phansalkar_p1, cmap=plt.cm.gray)
plt.title('Phansalkar Threshold, p=1.5')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.imshow(binary_phansalkar_p3, cmap=plt.cm.gray)
plt.title('Phansalkar Threshold, p=3.0')
plt.axis('off')

plt.show()
