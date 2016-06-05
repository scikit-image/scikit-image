"""
============
Thresholding
============

Thresholding is used to create a binary image from a grayscale image [1]_.
If you are not familiar with the details of the different algorithms and the
underlying assumptions, it is often to know which algorithm will give the best
results. Therefore, Scikit-image includes a function to test thresholding algorithms
provided in the library. At a glance, you can select the best algorithm
for you data, without a deep understanding of their mechanisms.

.. [1] https://en.wikipedia.org/wiki/Thresholding_%28image_processing%29

"""
import matplotlib
import matplotlib.pyplot as plt

from skimage import data
from skimage.filters import thresholding

img = data.page()

# Here, we specify a radius for local thresholding algorithm.
# If it is not specified, only global algorithms are called.
fig, ax = thresholding.try_all_threshold(img, radius=20,
                                         figsize=(10,8), verbose=False)
plt.show()

"""
.. image:: PLOT2RST.current_figure

How to apply a threshold?
=========================

Now, we illustrate how to apply one of these thresholding algorithms
This example uses the mean value of pixel intensities. It is a simple
and naive threshold value, which is sometimes used as a guess value.
"""

#from skimage.filters.thresholding import threshold_mean
#from skimage import data
#image = data.camera()
#thresh = threshold_mean(image)
#binary = image > thresh
#
#fig, axes = plt.subplots(nrows=2, figsize=(7, 8))
#ax0, ax1 = axes
#
#ax0.imshow(image)
#ax0.set_title('Original image')
#
#ax1.imshow(binary)
#ax1.set_title('Result')
#
#for ax in axes:
#    ax.axis('off')
#
#plt.show()

"""
.. image:: PLOT2RST.current_figure
"""
