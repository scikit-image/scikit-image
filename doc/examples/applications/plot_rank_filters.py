"""
===============================================================
Rank filters
===============================================================

Rank filters are non-linear filters using the local grey levels ordering to compute the filtered value.
This ensemble of filters share a common base: the local grey-level histogram extraction computed on
the neighborhood of a pixel (defined by a 2D structuring element).
If the filtered value is taken as the middle value of the histogram, we get the classical median filter.

Rank filters can be used for several purposes such as:

* image quality enhancement
  e.g. image smoothing, sharpening

* image pre-processing
  e.g. noise reduction, contrast enhancement

* feature extraction
  e.g. border detection, isolated point detection

* post-processing
  e.g. small object removal, object grouping, contour smoothing

Some well known filters are specific cases of rank filters [1]_ e.g. morphological dilation, morphological erosion,
median filters.

The different implementation availables in ``skimage`` are compared compare.

In this example, we will see how to filter a grey level image using some of the linear and non-linear filters
availables in skimage.  We use the ``camera`` image from ``skimage.data``.

.. [1] Pierre Soille, On morphological operators based on rank filters, Pattern Recognition 35 (2002) 527-535.
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import data

ima = data.camera()
hist = np.histogram(ima, bins=np.arange(0, 256))

plt.figure(figsize=(8, 3))
plt.subplot(121)
plt.imshow(ima, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.subplot(122)
plt.plot(hist[1][:-1], hist[0], lw=2)
plt.title('histogram of grey values')

"""
.. image:: PLOT2RST.current_figure

Noise removal
==============

some noise is added to the image, 1% of pixels are randomly set to 255, %1% are randomly set to 0.
The **median** filter is applied to remove the noise.

.. note:: there is different implementations of median filter : ``skimage.filter.median_filter``and
`skimage.filter.rank.median``
"""

noise = np.random.random(ima.shape)
nima = data.camera()
nima[noise>.99] = 255
nima[noise<.01] = 0

from skimage.filter.rank import median
from skimage.morphology import disk

fig = plt.figure(figsize=[10,7])

lo = median(nima,disk(1))
hi = median(nima,disk(5))
ext = median(nima,disk(20))
plt.subplot(2,2,1)
plt.imshow(nima,cmap=plt.cm.gray,vmin=0,vmax=255)
plt.xlabel('noised image')
plt.subplot(2,2,2)
plt.imshow(lo,cmap=plt.cm.gray,vmin=0,vmax=255)
plt.xlabel('median $r=1$')
plt.subplot(2,2,3)
plt.imshow(hi,cmap=plt.cm.gray,vmin=0,vmax=255)
plt.xlabel('median $r=5$')
plt.subplot(2,2,4)
plt.imshow(ext,cmap=plt.cm.gray,vmin=0,vmax=255)
plt.xlabel('median $r=20$')

"""
.. image:: PLOT2RST.current_figure

The added noise is efficiently removed, as the image defaults are small (1 pixel wide), a small filter radius is
sufficient. As the radius is increasing, objects with a bigger size are filtered too such as the camera tripod.
Median filter is commonly used for noise removal because borders are preserved.

Image smoothing
================

The example hereunder shows how a local **mean** smooth the cameraman image.

"""
from skimage.filter.rank import mean

fig = plt.figure(figsize=[10,7])

loc_mean = mean(nima,disk(10))
plt.subplot(1,2,1)
plt.imshow(ima,cmap=plt.cm.gray,vmin=0,vmax=255)
plt.xlabel('original')
plt.subplot(1,2,2)
plt.imshow(loc_mean,cmap=plt.cm.gray,vmin=0,vmax=255)
plt.xlabel('local mean $r=10$')

"""
.. image:: PLOT2RST.current_figure

One may be interested in smoothing an image while preserving important borders (median filters already achieved this),
here we use the **bilateral** filter that restrict the local neighborhood to pixel having a grey level similar to the
central one.

.. note:: a different implementations is available for color images in ``skimage.filter.denoise_bilateral``.

"""

from skimage.filter.rank import bilateral_mean

ima = data.camera()
selem = disk(10)

bilat = bilateral_mean(ima.astype(np.uint16),disk(20),s0=10,s1=10)

# display results
fig = plt.figure(figsize=[10,7])
plt.subplot(1,2,1)
plt.imshow(ima,cmap=plt.cm.gray)
plt.xlabel('original')
plt.subplot(1,2,2)
plt.imshow(bilat,cmap=plt.cm.gray)
plt.xlabel('bilateral mean')

"""
.. image:: PLOT2RST.current_figure

One can see that the large continuous part of the image (e.g.sky) are smoothed whereas other details are preserved.


Contrast enhancement
====================

We compare here how the global histogram equalization is applied locally.

The equalized image [2]_ has a roughly linear cumulative distribution function for each pixel neighborhood.
The local version [3]_ of the histogram equalization emphasized every local graylevel variations.

.. [2] http://en.wikipedia.org/wiki/Histogram_equalization
.. [3] http://en.wikipedia.org/wiki/Adaptive_histogram_equalization

"""

from skimage import exposure
from skimage.filter import rank

ima = data.camera()
# equalize globally and locally
glob = exposure.equalize(ima)*255
loc = rank.equalize(ima,disk(20))

# extract histogram for each image
hist = np.histogram(ima, bins=np.arange(0, 256))
glob_hist = np.histogram(glob, bins=np.arange(0, 256))
loc_hist = np.histogram(loc, bins=np.arange(0, 256))

plt.figure(figsize=(10, 10))
plt.subplot(321)
plt.imshow(ima, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.subplot(322)
plt.plot(hist[1][:-1], hist[0], lw=2)
plt.title('histogram of grey values')
plt.subplot(323)
plt.imshow(glob, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.subplot(324)
plt.plot(glob_hist[1][:-1], glob_hist[0], lw=2)
plt.title('histogram of grey values')
plt.subplot(325)
plt.imshow(loc, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.subplot(326)
plt.plot(loc_hist[1][:-1], loc_hist[0], lw=2)
plt.title('histogram of grey values')
"""
.. image:: PLOT2RST.current_figure

an other way to maximize the number of grey level used for an image is to apply a local autoleveling,
i.e. here a pixel grey level is proportionally remapped between local minimum and local maximum.

The following example show how local autolevel enhance the camaraman picture.
"""
from skimage.filter.rank import autolevel

ima = data.camera()
selem = disk(10)

auto = autolevel(ima.astype(np.uint16),disk(20))

# display results
fig = plt.figure(figsize=[10,7])
plt.subplot(1,2,1)
plt.imshow(ima,cmap=plt.cm.gray)
plt.xlabel('original')
plt.subplot(1,2,2)
plt.imshow(auto,cmap=plt.cm.gray)
plt.xlabel('local autolevel')
"""
.. image:: PLOT2RST.current_figure

This filter is very sensitive to local outlayers, see the little white spot in the sky left part. This is due
to a local maximum which is very high comparing to the rest of the neighborhood. One can moderate this
using the percentile version of the autolevel filter which uses to given percentiles (one inferior, one superior)
in place of local minimum and maximum. The example bellow illustrate how the percentile parameters influence the
local autolevel result.

"""
from skimage.filter.rank import percentile_autolevel

image = data.camera()

selem = disk(20)
loc_autolevel = autolevel(image,selem=selem)
loc_perc_autolevel0 = percentile_autolevel(image,selem=selem,p0=.00,p1=1.0)
loc_perc_autolevel1 = percentile_autolevel(image,selem=selem,p0=.01,p1=.99)
loc_perc_autolevel2 = percentile_autolevel(image,selem=selem,p0=.05,p1=.95)
loc_perc_autolevel3 = percentile_autolevel(image,selem=selem,p0=.1,p1=.9)

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax0, ax1, ax2 = axes
plt.gray()

ax0.imshow(np.hstack((image,loc_autolevel)))
ax0.set_title('original / autolevel')

ax1.imshow(np.hstack((loc_perc_autolevel0,loc_perc_autolevel1)),vmin=0,vmax=255)
ax1.set_title('percentile autolevel 0%,1%')
ax2.imshow(np.hstack((loc_perc_autolevel2,loc_perc_autolevel3)),vmin=0,vmax=255)
ax2.set_title('percentile autolevel 5% and 10%')

for ax in axes:
    ax.axis('off')

"""
.. image:: PLOT2RST.current_figure

Morphological contrast enhancement filter replaces the central pixel by local maximum
if the original grey level value if closest to local maximum, by the minimum local otherwise.

"""

from skimage.filter.rank import morph_contr_enh

ima = data.camera()

enh = morph_contr_enh(ima,disk(5))

# display results
fig = plt.figure(figsize=[10,7])
plt.subplot(1,2,1)
plt.imshow(ima,cmap=plt.cm.gray)
plt.xlabel('original')
plt.subplot(1,2,2)
plt.imshow(enh,cmap=plt.cm.gray)
plt.xlabel('local morphlogical contrast enhancement')

"""
.. image:: PLOT2RST.current_figure

The percentile version of the local morphological contrast enhancement, uses percentile p0 and p1 instead of local
minimum and local maximum.

"""

from skimage.filter.rank import percentile_morph_contr_enh

ima = data.camera()

penh = percentile_morph_contr_enh(ima,disk(5),p0=.1,p1=.9)

# display results
fig = plt.figure(figsize=[10,7])
plt.subplot(1,2,1)
plt.imshow(ima,cmap=plt.cm.gray)
plt.xlabel('original')
plt.subplot(1,2,2)
plt.imshow(penh,cmap=plt.cm.gray)
plt.xlabel('local morphlogical contrast enhancement')

"""
.. image:: PLOT2RST.current_figure

Image morphology
================

Local maximum and local minimum are the base operators for grey level morphology.

"""

"""
.. image:: PLOT2RST.current_figure

Implementation
================

Implementation comparison w.r.t. image size and structuring element size.

"""
plt.show()
