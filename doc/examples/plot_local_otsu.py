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
import matplotlib.pyplot as plt

from skimage import data
from skimage.morphology import disk
from skimage.filter import threshold_otsu, rank
from skimage.util import img_as_ubyte


p8 = img_as_ubyte(data.page())

radius = 10
selem = disk(radius)

loc_otsu = rank.otsu(p8, selem)
t_glob_otsu = threshold_otsu(p8)
glob_otsu = p8 >= t_glob_otsu


plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(p8, cmap=plt.cm.gray)
plt.xlabel('original')
plt.colorbar()
plt.subplot(2, 2, 2)
plt.imshow(loc_otsu, cmap=plt.cm.gray)
plt.xlabel('local Otsu ($radius=%d$)' % radius)
plt.colorbar()
plt.subplot(2, 2, 3)
plt.imshow(p8 >= loc_otsu, cmap=plt.cm.gray)
plt.xlabel('original >= local Otsu' % t_glob_otsu)
plt.subplot(2, 2, 4)
plt.imshow(glob_otsu, cmap=plt.cm.gray)
plt.xlabel('global Otsu ($t = %d$)' % t_glob_otsu)
plt.show()
