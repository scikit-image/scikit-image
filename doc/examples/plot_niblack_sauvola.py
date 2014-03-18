"""
================================
Niblack and Sauvola Thresholding
================================
Niblack and Sauvola thresholds are local thresholding techniques that
are useful for images where the background is not uniform. Instead of
calculating a single global threshold for the entire image, several
thresholds are calculated for every pixel by using specific formulas
that take into account the mean and standard deviation of the local
neighborhood (defined by a window w x w centered around the pixel).

Here, we binarize an image using `threshold_niblack` and
`threshold_sauvola` and compare it to a common global thresholding
technique. Parameter `w` determines the size of the window that
contains the surrounding pixels.

.. [1] Niblack, W (1986), An introduction to Digital Image
       Processing, Prentice-Hall.
.. [2] J. Sauvola and M. Pietikainen, "Adaptive document image
       binarization," Pattern Recognition 33(2),
       pp. 225-236, 2000.
.. [3] C. Wolf, J-M. Jolion, "Extraction and Recognition of
       Artificial Text in Multimedia Documents", Pattern
       Analysis and Applications, 6(4):309-326, (2003).
.. [4] Phansalskar, N; More, S & Sabale, A et al. (2011), "Adaptive
       local thresholding for detection of nuclei in diversity
       stained cytology images.", International Conference on
       Communications and Signal Processing (ICCSP): 218-220.
"""
import matplotlib
import matplotlib.pyplot as plt

from skimage.data import page
from skimage.filter import (threshold_otsu, threshold_niblack,
                            threshold_sauvola)


matplotlib.rcParams['font.size'] = 9


image = page()
binary_global = image > threshold_otsu(image)

window_size = 25
binary_niblack = threshold_niblack(image, w=window_size, k=0.8)
binary_sauvola = threshold_sauvola(image, w=window_size)  # Original Sauvola
binary_wolf = threshold_sauvola(image, w=window_size, method='wolf', k=0.5)
binary_phansalkar = threshold_sauvola(image, w=window_size,
                                      method='phansalkar', r=50)
# Parameters k, r, etc. are other parameters that directly affect
# threshold calculation formula. Check each method's documentation to see the
# respective definition of its threshold formula.

plt.figure(figsize=(8, 7))
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
plt.imshow(binary_wolf, cmap=plt.cm.gray)
plt.title('Sauvola Threshold (Wolf)')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.imshow(binary_phansalkar, cmap=plt.cm.gray)
plt.title('Sauvola Threshold (Phansalkar)')
plt.axis('off')

plt.show()
