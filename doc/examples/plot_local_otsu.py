"""
====================
Local Otsu Threshold
====================

This example shows how Otsu's threshold [1]_ method can be applied locally. For
each pixel, an "optimal" threshold is determined by maximizing the variance
between two classes of pixels of the local neighborhood defined by a structuring
element.

The example compares the local threshold with the global threshold.

.. note: local is much slower than global thresholding

.. [1] http://en.wikipedia.org/wiki/Otsu's_method

"""
import matplotlib
import matplotlib.pyplot as plt

from skimage import data
from skimage.morphology import disk
from skimage.filter import threshold_otsu, rank
from skimage.util import img_as_ubyte


matplotlib.rcParams['font.size'] = 9


img = img_as_ubyte(data.page())

radius = 15
selem = disk(radius)

local_otsu = rank.otsu(img, selem)
threshold_global_otsu = threshold_otsu(img)
global_otsu = img >= threshold_global_otsu


plt.figure(figsize=(8, 5))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap=plt.cm.gray)
plt.title('Original')
plt.colorbar(orientation='horizontal')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(local_otsu, cmap=plt.cm.gray)
plt.title('Local Otsu (radius=%d)' % radius)
plt.colorbar(orientation='horizontal')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(img >= local_otsu, cmap=plt.cm.gray)
plt.title('Original >= Local Otsu' % threshold_global_otsu)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(global_otsu, cmap=plt.cm.gray)
plt.title('Global Otsu (threshold = %d)' % threshold_global_otsu)
plt.axis('off')

plt.show()
