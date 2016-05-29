"""
============
Thresholding
============

Thresholding is used to create a binary image from a grayscale image [1]_.

Thresholding algorithms can be separated in two categories:
    * Global. They are based on the histogram of the pixel intensity of
    the image.
    * Local. To process a pixel, only the neighboring pixels are used.
    These algorithms often require more computation time.

Scikit-image includes a function to test thresholding algorithms provided
in the library. Therefore, in a glance, you can select the best algorithm
for you data, without a deep understanding of their mechanisms.

.. [1] https://en.wikipedia.org/wiki/Thresholding_%28image_processing%29

"""
from skimage.data import page

img = page()

# Here, we specify a radius for local thresholding algorithm.
# If it is not specified, only global algorithms are called.
fig, ax = mosaic_threshold(img, radius=20, figsize=(10,8), verbose=False)
fig

"""

.. image:: PLOT2RST.current_figure

This example uses Otsu's method [2]_ to calculate the threshold value.

Otsu's method calculates an "optimal" threshold (marked by a red line in the
histogram below) by maximizing the variance between two classes of pixels,
which are separated by the threshold. Equivalently, this threshold minimizes
the intra-class variance.

.. [2] http://en.wikipedia.org/wiki/Otsu's_method

"""
import matplotlib
import matplotlib.pyplot as plt

from skimage.data import camera
from skimage.filters import threshold_otsu


matplotlib.rcParams['font.size'] = 9


image = camera()
thresh = threshold_otsu(image)
binary = image > thresh

#fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.5))
fig = plt.figure(figsize=(8, 2.5))
ax1 = plt.subplot(1, 3, 1, adjustable='box-forced')
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1, adjustable='box-forced')

ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Original')
ax1.axis('off')

ax2.hist(image)
ax2.set_title('Histogram')
ax2.axvline(thresh, color='r')

ax3.imshow(binary, cmap=plt.cm.gray)
ax3.set_title('Thresholded')
ax3.axis('off')

plt.show()
