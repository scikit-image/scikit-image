"""
============
Thresholding
============

Thresholding is used to create a binary image from a grayscale image [1]_.

.. [1] https://en.wikipedia.org/wiki/Thresholding_%28image_processing%29

.. seealso::
    A more comprehensive presentation on
    :ref:`sphx_glr_auto_examples_xx_applications_plot_thresholding.py`

"""

######################################################################
# We illustrate how to apply one of these thresholding algorithms.
# This example uses the mean value of pixel intensities. It is a simple
# and naive threshold value, which is sometimes used as a guess value.

import matplotlib.pyplot as plt
from skimage.filters.thresholding import threshold_mean
from skimage import data

image = data.camera()
thresh = threshold_mean(image)
binary = image > thresh

fig, axes = plt.subplots(ncols=2, figsize=(8, 3))
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(binary, cmap=plt.cm.gray)
ax[1].set_title('Result')

for a in ax:
    a.axis('off')

plt.show()


######################################################################
# If you are not familiar with the details of the different algorithms and the
# underlying assumptions, it is often difficult to know which algorithm will give
# the best results. Therefore, Scikit-image includes a function to evaluate
# thresholding algorithms provided by the library. At a glance, you can select
# the best algorithm for you data without a deep understanding of their
# mechanisms.

from skimage.filters import thresholding

img = data.page()

# Here, we specify a radius for local thresholding algorithms.
# If it is not specified, only global algorithms are called.
fig, ax = thresholding.try_all_threshold(img, radius=20,
                                         figsize=(10, 8), verbose=False)
plt.show()
