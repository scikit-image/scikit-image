"""
============
Thresholding
============

Thresholding is used to create a binary image from a grayscale image [1]_.
It is the simplest way to segment objects from a background.

Thresholding algorithms implemented in scikit-image can be separated in two
categories:

- Histogram-based. The histogram of the pixel intensity is used and
  assumptions may be made on the properties of this histogram (e.g. bimodal).
- Local. To process a pixel, only the neighboring pixels are used.
  These algorithms often require more computation time.


If you are not familiar with the details of the different algorithms and the
underlying assumptions, it is often to know which algorithm will give the best
results. Therefore, Scikit-image includes a function to test thresholding
algorithms provided in the library. At a glance, you can select the best
algorithm for you data, without a deep understanding of their mechanisms.

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
                                         figsize=(10, 8), verbose=False)
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

Bimodal histogram
=================

For pictures with a bimodal histogram, more specific algorithms can be used.
For instance, the minimum algorithm takes a histogram of the image and smooths it
repeatedly until there are only two peaks in the histogram.  Then it
finds the minimum value between the two peaks.  After smoothing the
histogram, there can be multiple pixel values with the minimum histogram
count, so you can pick the 'min', 'mid', or 'max' of these values.

"""
import matplotlib.pyplot as plt

from skimage import data
from skimage.filters.thresholding import threshold_minimum

image = data.camera()

thresh = threshold_minimum(image, bias='min')
binary = image > thresh

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

"""

.. image:: PLOT2RST.current_figure

Otsu's method [2]_ calculates an "optimal" threshold (marked by a red line in the
histogram below) by maximizing the variance between two classes of pixels,
which are separated by the threshold. Equivalently, this threshold minimizes
the intra-class variance.

.. [2] http://en.wikipedia.org/wiki/Otsu's_method

"""
import matplotlib
import matplotlib.pyplot as plt

from skimage import data
from skimage.filters import threshold_otsu


matplotlib.rcParams['font.size'] = 9


image = data.camera()
thresh = threshold_otsu(image)
binary = image > thresh

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

"""
.. image:: PLOT2RST.current_figure

Local thresholding
==================

If the image background is relatively uniform, then you can use a global
threshold value as presented above. However, if there is large variation in the
background intensity, adaptive thresholding (a.k.a. local or dynamic
thresholding) may produce better results. Note that local is much slower than
global thresholding

Here, we binarize an image using the `threshold_adaptive` function, which
calculates thresholds in regions of size `block_size` surrounding each pixel
(i.e. local neighborhoods). Each threshold value is the weighted mean of the
local neighborhood minus an offset value.

"""
import matplotlib.pyplot as plt

from skimage import data
from skimage.filters import threshold_otsu, threshold_adaptive


image = data.page()

global_thresh = threshold_otsu(image)
binary_global = image > global_thresh

block_size = 35
binary_adaptive = threshold_adaptive(image, block_size, offset=10)

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax0, ax1, ax2 = axes
plt.gray()

ax0.imshow(image)
ax0.set_title('Original')

ax1.imshow(binary_global)
ax1.set_title('Global thresholding')

ax2.imshow(binary_adaptive)
ax2.set_title('Adaptive thresholding')

for ax in axes:
    ax.axis('off')

plt.show()

"""
.. image:: PLOT2RST.current_figure

Now, we show how Otsu's threshold [2]_ method can be applied locally. For
each pixel, an "optimal" threshold is determined by maximizing the variance
between two classes of pixels of the local neighborhood defined by a
structuring element.

The example compares the local threshold with the global threshold.

"""

from skimage import data
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.size'] = 9
img = img_as_ubyte(data.page())

radius = 15
selem = disk(radius)

local_otsu = rank.otsu(img, selem)
threshold_global_otsu = threshold_otsu(img)
global_otsu = img >= threshold_global_otsu

fig, ax = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})
ax0, ax1, ax2, ax3 = ax.ravel()
plt.tight_layout()

fig.colorbar(ax0.imshow(img, cmap=plt.cm.gray),
             ax=ax0, orientation='horizontal')
ax0.set_title('Original')
ax0.axis('off')

fig.colorbar(ax1.imshow(local_otsu, cmap=plt.cm.gray),
             ax=ax1, orientation='horizontal')
ax1.set_title('Local Otsu (radius=%d)' % radius)
ax1.axis('off')

ax2.imshow(img >= local_otsu, cmap=plt.cm.gray)
ax2.set_title('Original >= Local Otsu' % threshold_global_otsu)
ax2.axis('off')

ax3.imshow(global_otsu, cmap=plt.cm.gray)
ax3.set_title('Global Otsu (threshold = %d)' % threshold_global_otsu)
ax3.axis('off')

plt.show()
