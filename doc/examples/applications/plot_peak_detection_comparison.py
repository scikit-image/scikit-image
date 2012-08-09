"""
==============
Peak detection
==============

Peak detection (a.k.a. spot detection or particle detection) is a common
image analysis step. For example, it's used to detect tracer particles in a
flow for particle image velocimetry (i.e. PIV) and to identify features in
the Hough transform.

To simplify plotting code, let's define a simple function that creates a new
figure on each call, and removes tick labels.
"""

import matplotlib.pyplot as plt

plt.rcParams['axes.titlesize'] = 10
plt.rcParams['font.size'] = 10
def imshow(image, **kwargs):
    plt.figure(figsize=(2.5, 2.5))
    plt.imshow(image, **kwargs)
    plt.axis('off')

"""
To explore different peak detection techniques, we use an image of circles with
added noise:
"""

from skimage import data
img = data.load('noisy_circles.jpg')
imshow(img)

"""

.. image:: PLOT2RST.current_figure

This image is noisy and has uneven background illumination. The peaks in the
image, while readily identified by eye, can be tricky to find algorithmically.
The first thing we need to do is remove the high-frequency noise; this can
be done with a simple Gaussian filter.
"""

import scipy.ndimage as ndimg
img_smooth = ndimg.gaussian_filter(img, 3)

imshow(img_smooth)

"""

.. image:: PLOT2RST.current_figure

Thresholding
============

One way to extract the background is to threshold the image.
"""

thresh_value = 100
background = img_smooth.copy()
background[img_smooth > thresh_value] = 0
peaks = img_smooth - background

"""
Here, all pixels values below the threshold value are subtracted from the
image. The resulting background image and the extracted peaks are shown below.
"""

imshow(background, vmin=0, vmax=255)
plt.title("background image (thresholding)")

"""
.. image:: PLOT2RST.current_figure
"""

imshow(peaks, vmin=0, vmax=255)
plt.title("peaks (thresholding)")

"""
.. image:: PLOT2RST.current_figure

Because of uneven illumination, peaks on the right bleed into each other.
Increasing the threshold will fix this problem, but it will also cause some
peaks on the left to go undetected.

Morphological reconstruction
============================
"""

import numpy as np
img_r = np.int32(img_smooth)

import skimage.morphology as morph
h = 20
rec = morph.reconstruction(img_r-h, img_r)

imshow(img_r, vmin=0, vmax=255)
plt.title("original (smoothed) image")

"""
.. image:: PLOT2RST.current_figure
"""

imshow(rec, vmin=0, vmax=255)
plt.title("background image (reconstruction)")

"""
.. image:: PLOT2RST.current_figure

This reconstructed image looks pretty much like the original, except that the
peaks in the image are truncated. The reconstructed image can then be
subtracted from the original image to reveal the peaks of the image.
"""

imshow(img_r-rec)
plt.title("h-dome of image")

"""
.. image:: PLOT2RST.current_figure

The result is known as the h-dome transformation [2]_, which extracts peaks of
height `h` from the original image. To better understand what's going on,
let's take a 1D slice along the middle of the image (cutting through peaks in
the image).
"""

img_slice = img_r[99:100, :]
rec_slice = morph.reconstruction(img_slice-h, img_slice)

"""
Plotting the reconstructed image (slice) next to the original image and the
seed image shed light on the reconstruction process
"""
plt.figure(figsize=(4, 3))
plt.plot(img_slice[0], 'k', label='original image')
plt.plot(img_slice[0]-h, '0.5', label='seed image')
plt.plot(rec_slice[0], 'r', label='reconstructed')
plt.title("image slice")
plt.xlabel('x')
plt.ylabel('intensity')
plt.legend()

"""
.. image:: PLOT2RST.current_figure

Here, you see that morphological reconstruction dilates the seed image (i.e.
the `h`-shifted image) until it intersects the mask (original image). Note that
the peaks in the original image have very different intensity values (e.g. the
peak at x=200 and x=100 differ by about 80). Subtracting the reconstructed
image from the original image gives peaks of roughly equal intensity. Thus, the
h-dome transformation is quiet effective at removing uneven, dark backgrounds
from bright features. The inverse operation---the h-basin
transformation---should be used when removing bright backgrounds from dark
features.


White tophat
============
"""
selem = morph.disk(10)
img_t = np.uint8(img_smooth)
opening = morph.opening(img_t, selem)
top_hat = img_t - opening

imshow(opening, vmin=0, vmax=255)
plt.title("Greyscale opening of image")

"""
.. image:: PLOT2RST.current_figure
"""


imshow(top_hat)
plt.title("Tophat with disk of r = 10")

"""
.. image:: PLOT2RST.current_figure
"""

selem = morph.disk(5)
top_hat = morph.white_tophat(img_t, selem)

imshow(top_hat)
plt.title("Tophat with disk of r = 5")

"""
.. image:: PLOT2RST.current_figure
"""

selem = morph.square(20)
opening = morph.opening(img_t, selem)
# scikit's top hat filter uses uint8 and doesn't check for over(under)flow.
mask = opening > img_t
opening[mask] = img_t[mask]
top_hat = img_t - opening

imshow(opening, vmin=0, vmax=255)
plt.title("Greyscale opening of image")

"""
.. image:: PLOT2RST.current_figure
"""

imshow(top_hat)
plt.title("Tophat with square of w = 10")

plt.show()

"""
.. image:: PLOT2RST.current_figure


References
==========

.. [1] Crocker and Grier, Journal of Colloid and Interface Science (1996)
.. [2] Vincent, L., IEEE Transactions on Image Processing (1993)

"""
