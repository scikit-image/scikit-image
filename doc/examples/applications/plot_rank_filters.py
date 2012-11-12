"""
============
Rank filters
============

Rank filters are non-linear filters using the local grey levels ordering to
compute the filtered value. This ensemble of filters share a common base: the
local grey-level histogram extraction computed on the neighborhood of a pixel
(defined by a 2D structuring element). If the filtered value is taken as the
middle value of the histogram, we get the classical median filter.

Rank filters can be used for several purposes such as:

* image quality enhancement
  e.g. image smoothing, sharpening

* image pre-processing
  e.g. noise reduction, contrast enhancement

* feature extraction
  e.g. border detection, isolated point detection

* post-processing
  e.g. small object removal, object grouping, contour smoothing

Some well known filters are specific cases of rank filters [1]_ e.g.
morphological dilation, morphological erosion, median filters.

The different implementation availables in `skimage` are compared.

In this example, we will see how to filter a grey level image using some of the
linear and non-linear filters availables in skimage.  We use the `camera`
image from `skimage.data`.

.. [1] Pierre Soille, On morphological operators based on rank filters, Pattern
       Recognition 35 (2002) 527-535.

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
=============

Some noise is added to the image, 1% of pixels are randomly set to 255, 1% are
randomly set to 0. The **median** filter is applied to remove the noise.

.. note::

    there are different implementations of median filter :
    `skimage.filter.median_filter` and `skimage.filter.rank.median`

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

The added noise is efficiently removed, as the image defaults are small (1 pixel
wide), a small filter radius is sufficient. As the radius is increasing, objects
with a bigger size are filtered as well, such as the camera tripod. The median
filter is commonly used for noise removal because borders are preserved.

Image smoothing
================

The example hereunder shows how a local **mean** smoothes the camera man image.

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

One may be interested in smoothing an image while preserving important borders
(median filters already achieved this), here we use the **bilateral** filter
that restricts the local neighborhood to pixel having a grey level similar to
the central one.

.. note::

    a different implementation is available for color images in
    `skimage.filter.denoise_bilateral`.

"""

from skimage.filter.rank import bilateral_mean

ima = data.camera()
selem = disk(10)

bilat = bilateral_mean(ima.astype(np.uint16),disk(20),s0=10,s1=10)

# display results
fig = plt.figure(figsize=[10,7])
plt.subplot(2,2,1)
plt.imshow(ima,cmap=plt.cm.gray)
plt.xlabel('original')
plt.subplot(2,2,3)
plt.imshow(bilat,cmap=plt.cm.gray)
plt.xlabel('bilateral mean')
plt.subplot(2,2,2)
plt.imshow(ima[200:350,350:450],cmap=plt.cm.gray)
plt.subplot(2,2,4)
plt.imshow(bilat[200:350,350:450],cmap=plt.cm.gray)

"""

.. image:: PLOT2RST.current_figure

One can see that the large continuous part of the image (e.g. sky) is smoothed
whereas other details are preserved.


Contrast enhancement
====================

We compare here how the global histogram equalization is applied locally.

The equalized image [2]_ has a roughly linear cumulative distribution function
for each pixel neighborhood. The local version [3]_ of the histogram
equalization emphasizes every local graylevel variations.

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

another way to maximize the number of grey levels used for an image is to apply
a local autoleveling, i.e. here a pixel grey level is proportionally remapped
between local minimum and local maximum.

The following example shows how local autolevel enhances the camara man picture.

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

This filter is very sensitive to local outlayers, see the little white spot in
the sky left part. This is due to a local maximum which is very high comparing
to the rest of the neighborhood. One can moderate this using the percentile
version of the autolevel filter which uses given percentiles (one inferior,
one superior) in place of local minimum and maximum. The example below
illustrates how the percentile parameters influence the local autolevel result.

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

The morphological contrast enhancement filter replaces the central pixel by the
local maximum if the original pixel value is closest to local maximum, otherwise
by the minimum local.

"""

from skimage.filter.rank import morph_contr_enh

ima = data.camera()

enh = morph_contr_enh(ima,disk(5))

# display results
fig = plt.figure(figsize=[10,7])
plt.subplot(2,2,1)
plt.imshow(ima,cmap=plt.cm.gray)
plt.xlabel('original')
plt.subplot(2,2,3)
plt.imshow(enh,cmap=plt.cm.gray)
plt.xlabel('local morphlogical contrast enhancement')
plt.subplot(2,2,2)
plt.imshow(ima[200:350,350:450],cmap=plt.cm.gray)
plt.subplot(2,2,4)
plt.imshow(enh[200:350,350:450],cmap=plt.cm.gray)

"""

.. image:: PLOT2RST.current_figure

The percentile version of the local morphological contrast enhancement uses
percentile *p0* and *p1* instead of the local minimum and maximum.

"""

from skimage.filter.rank import percentile_morph_contr_enh

ima = data.camera()

penh = percentile_morph_contr_enh(ima,disk(5),p0=.1,p1=.9)

# display results
fig = plt.figure(figsize=[10,7])
plt.subplot(2,2,1)
plt.imshow(ima,cmap=plt.cm.gray)
plt.xlabel('original')
plt.subplot(2,2,3)
plt.imshow(penh,cmap=plt.cm.gray)
plt.xlabel('local percentile morphlogical\n contrast enhancement')
plt.subplot(2,2,2)
plt.imshow(ima[200:350,350:450],cmap=plt.cm.gray)
plt.subplot(2,2,4)
plt.imshow(penh[200:350,350:450],cmap=plt.cm.gray)

"""

.. image:: PLOT2RST.current_figure

Image threshold
===============

The Otsu's threshold [1]_ method can be applied locally using the local
greylevel distribution. In the example below, for each pixel, an "optimal"
threshold is determined by maximizing the variance between two classes of pixels
of the local neighborhood defined by a structuring element.

The example compares the local threshold with the global threshold
`skimage.filter.threshold_otsu`.

.. note::

    Local thresholding is much slower than global one. There exists a function
    for global Otsu thresholding: `skimage.filter.threshold_otsu`.

.. [1] http://en.wikipedia.org/wiki/Otsu's_method

"""

from skimage.filter.rank import otsu
from skimage.filter import threshold_otsu

p8 = data.page()

radius = 10
selem = disk(radius)

# t_loc_otsu is an image
t_loc_otsu = otsu(p8,selem)
loc_otsu = p8>=t_loc_otsu

# t_glob_otsu is a scalar
t_glob_otsu = threshold_otsu(p8)
glob_otsu = p8>=t_glob_otsu

plt.figure()
plt.subplot(2,2,1)
plt.imshow(p8,cmap=plt.cm.gray)
plt.xlabel('original')
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(t_loc_otsu,cmap=plt.cm.gray)
plt.xlabel('local Otsu ($radius=%d$)'%radius)
plt.colorbar()
plt.subplot(2,2,3)
plt.imshow(p8>=t_loc_otsu,cmap=plt.cm.gray)
plt.xlabel('original>=local Otsu'%t_glob_otsu)
plt.subplot(2,2,4)
plt.imshow(glob_otsu,cmap=plt.cm.gray)
plt.xlabel('global Otsu ($t=%d$)'%t_glob_otsu)

"""

.. image:: PLOT2RST.current_figure

The following example shows how local Otsu's threshold handles a global level
shift applied to a synthetic image .

"""

n = 100
theta = np.linspace(0,10*np.pi,n)
x = np.sin(theta)
m = (np.tile(x,(n,1))* np.linspace(0.1,1,n)*128+128).astype(np.uint8)

radius = 10
t = rank.otsu(m,disk(radius))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(m)
plt.xlabel('original')
plt.subplot(1,2,2)
plt.imshow(m>=t,interpolation='nearest')
plt.xlabel('local Otsu ($radius=%d$)'%radius)

"""

.. image:: PLOT2RST.current_figure

Image morphology
================

Local maximum and local minimum are the base operators for grey level
morphology.

.. note::

    `skimage.dilate` and `skimage.erode` are equivalent filters (see below for
    comparison).

Here is an example of the classical morphological grey level filters: opening,
closing and morphological gradient.

"""

from skimage.filter.rank import maximum,minimum,gradient

ima = data.camera()

closing = maximum(minimum(ima,disk(5)),disk(5))
opening = minimum(maximum(ima,disk(5)),disk(5))
grad = gradient(ima,disk(5))

# display results
fig = plt.figure(figsize=[10,7])
plt.subplot(2,2,1)
plt.imshow(ima,cmap=plt.cm.gray)
plt.xlabel('original')
plt.subplot(2,2,2)
plt.imshow(closing,cmap=plt.cm.gray)
plt.xlabel('grey level closing')
plt.subplot(2,2,3)
plt.imshow(opening,cmap=plt.cm.gray)
plt.xlabel('grey level opening')
plt.subplot(2,2,4)
plt.imshow(grad,cmap=plt.cm.gray)
plt.xlabel('morphological gradient')

"""

.. image:: PLOT2RST.current_figure

Feature extraction
===================

Local histogram can be exploited to compute local entropy, which is related to
the local image complexity. Entropy is computed using base 2 logarithm i.e. the
filter returns the minimum number of bits needed to encode local greylevel
distribution.

`skimage.rank.entropy` returns local entropy on a given structuring element.
The following example shows this filter applied on 8- and 16- bit images.

.. note::

    to better use the available image bit, the function returns 10x entropy for
    8-bit images and 1000x entropy for 16-bit images.

"""

from skimage import data
from skimage.filter.rank import entropy
from skimage.morphology import disk
import numpy as np
import matplotlib.pyplot as plt

# defining a 8- and a 16-bit test images
a8 = data.camera()
a16 = data.camera().astype(np.uint16)*4

ent8 = entropy(a8,disk(5)) # pixel value contain 10x the local entropy
ent16 = entropy(a16,disk(5)) # pixel value contain 1000x the local entropy

# display results
plt.figure(figsize=(10, 10))

plt.subplot(2,2,1)
plt.imshow(a8, cmap=plt.cm.gray)
plt.xlabel('8-bit image')
plt.colorbar()

plt.subplot(2,2,2)
plt.imshow(ent8, cmap=plt.cm.jet)
plt.xlabel('entropy*10')
plt.colorbar()

plt.subplot(2,2,3)
plt.imshow(a16, cmap=plt.cm.gray)
plt.xlabel('16-bit image')
plt.colorbar()

plt.subplot(2,2,4)
plt.imshow(ent16, cmap=plt.cm.jet)
plt.xlabel('entropy*1000')
plt.colorbar()

"""

.. image:: PLOT2RST.current_figure

Implementation
================

The central part of the `skimage.rank` filters is build on a sliding window that
update local grey level histogram. This approach limits the algorithm complexity
to O(n) where n is the number of image pixels. The complexity is also limited
with respect to the structuring element size.

"""

from time import time

from scipy.ndimage.filters import percentile_filter
from skimage.morphology import dilation
from skimage.filter import median_filter
from skimage.filter.rank import median,maximum

def exec_and_timeit(func):
    """Decorator that returns both function results and execution time."""
    def wrapper(*arg):
        t1 = time()
        res = func(*arg)
        t2 = time()
        ms = (t2-t1)*1000.0
        return (res,ms)
    return wrapper


@exec_and_timeit
def cr_med(image,selem):
    return median(image=image,selem = selem)

@exec_and_timeit
def cr_max(image,selem):
    return maximum(image=image,selem = selem)

@exec_and_timeit
def cm_dil(image,selem):
    return dilation(image=image,selem = selem)

@exec_and_timeit
def ctmf_med(image,radius):
    return median_filter(image=image,radius=radius)

@exec_and_timeit
def ndi_med(image,n):
    return percentile_filter(image,50,size=n*2-1)

"""

Comparison between

* `rank.maximum`
* `cmorph.dilate`

on increasing structuring element size

"""

a = data.camera()

rec = []
e_range = range(1,20,2)
for r in e_range:
    elem = disk(r+1)
    rc,ms_rc = cr_max(a,elem)
    rcm,ms_rcm = cm_dil(a,elem)
    rec.append((ms_rc,ms_rcm))

rec = np.asarray(rec)

plt.figure()
plt.title('increasing element size')
plt.ylabel('time (ms)')
plt.xlabel('element radius')
plt.plot(e_range,rec)
plt.legend(['crank.maximum','cmorph.dilate'])

"""

and increasing image size

.. image:: PLOT2RST.current_figure

"""

r = 9
elem = disk(r+1)

rec = []
s_range = range(100,1000,100)
for s in s_range:
    a = (np.random.random((s,s))*256).astype('uint8')
    (rc,ms_rc) = cr_max(a,elem)
    (rcm,ms_rcm) = cm_dil(a,elem)
    rec.append((ms_rc,ms_rcm))

rec = np.asarray(rec)

plt.figure()
plt.title('increasing image size')
plt.ylabel('time (ms)')
plt.xlabel('image size')
plt.plot(s_range,rec)
plt.legend(['crank.maximum','cmorph.dilate'])


"""

.. image:: PLOT2RST.current_figure

Comparison between:

* `rank.median`
* `ctmf.median_filter`
* `ndimage.percentile`

on increasing structuring element size

"""

a = data.camera()

rec = []
e_range = range(2,30,4)
for r in e_range:
    elem = disk(r+1)
    rc,ms_rc = cr_med(a,elem)
    rctmf,ms_rctmf = ctmf_med(a,r)
    rndi,ms_ndi = ndi_med(a,r)
    rec.append((ms_rc,ms_rctmf,ms_ndi))

rec = np.asarray(rec)

plt.figure()
plt.title('increasing element size')
plt.plot(e_range,rec)
plt.legend(['rank.median','ctmf.median_filter','ndimage.percentile'])
plt.ylabel('time (ms)')
plt.xlabel('element radius')

"""
.. image:: PLOT2RST.current_figure

comparison of outcome of the three methods

"""

plt.figure()
plt.imshow(np.hstack((rc,rctmf,rndi)))
plt.xlabel('rank.median vs ctmf.median_filter vs ndimage.percentile')

"""
.. image:: PLOT2RST.current_figure

and increasing image size

"""

r = 9
elem = disk(r+1)

rec = []
s_range = [100,200,500,1000]
for s in s_range:
    a = (np.random.random((s,s))*256).astype('uint8')
    (rc,ms_rc) = cr_med(a,elem)
    rctmf,ms_rctmf = ctmf_med(a,r)
    rndi,ms_ndi = ndi_med(a,r)
    rec.append((ms_rc,ms_rctmf,ms_ndi))

rec = np.asarray(rec)

plt.figure()
plt.title('increasing image size')
plt.plot(s_range,rec)
plt.legend(['rank.median','ctmf.median_filter','ndimage.percentile'])
plt.ylabel('time (ms)')
plt.xlabel('image size')

"""
.. image:: PLOT2RST.current_figure

"""

plt.show()
