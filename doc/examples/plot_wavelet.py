"""
=============================================
Wavelet Coefficient Visualization & Denoising
=============================================

This example shows the results of a simple denoising filters using the discrete
wavelet transform[1].

An n-level discrete wavelet transform decomposes a digital image into 1 + 3*n
coefficient matrices. The first is called the approximation sub-band, which
captures the mean (and other low-frequency information) of the original image,
and is essentially a down-sampled representation of it.

The remaining coefficient matrices are called the detail sub-bands. Thresholding
functions are applied to these matrices. The wavelet transform is then
reversed, to give the denoised result. Thresholding can either be soft or
hard[2].

The main variation between the wavelet denoising methods that are out there in
the literature are in how the thresholding values for each detail sub-band is
determined (whether it is uniform across all detail sub-bands, or individual
for each, etc).

The `wavelet_filter` function allows a user to specify the detail thresholds
either uniformly, or by level, or by individual sub-band. This way, this
function can be used to implement the majority of wavelet denoising methods
out there (BayesShrink. SureShrink. etc) by writing short wrapping functions
which pre-compute the thresholds accordingly.

Arguments to wavelet_filter are the image to be
denoised, threshold value(s), and all arguments which could be passed to
pywt.wavedec2. Note that in wavelet_filter, the wavelet function to use is an
optional argument, and defaults to 'haar'.

[1] http://en.wikipedia.org/wiki/Discrete_wavelet_transform
[2] Donoho, David L., and Jain M. Johnstone. "Ideal spatial adaptation by
    wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
"""
import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage.filter import (
    wavelet_filter, wavelet_coefficient_array, bayes_shrink, visu_shrink)

"""
First, we load an example image & project it to gray-scale
"""
img = data.lena().mean(axis=2)

f = plt.figure()
ax = f.add_subplot(111)
ax.set_axis_off()
plt.title("Exact")
plt.imshow(img, cmap=plt.cm.gray)
"""
.. image:: PLOT2RST.current_figure

Next, we add some noise
"""
m, n, = img.shape
noisy = img + 15 * np.random.randn(m, n)

f = plt.figure()
ax = f.add_subplot(111)
ax.set_axis_off()
plt.title("Noisy")
plt.imshow(noisy, cmap=plt.cm.gray)

"""
.. image:: PLOT2RST.current_figure

Before denoising, let's visualize the difference in wavelet transform domain
between the original image, and the one contaminated with noise.
We can use the `wavelet_coefficient_array` function to give us all of the
wavelet coefficient sub-bands at a specified level, packed into a single
array for easy viewing.

For this, we'll simply use the default wavelet ('haar'), since it tends to
give wavelet coefficients bands with sharp edges and good contrast (which is
helpful for visualization). We'll also use the default number of levels (one),
which leads to the images themselves being the only necessary input parameters.
"""

cExact = wavelet_coefficient_array(img)
cNoisy = wavelet_coefficient_array(noisy)
plt.figure()
plt.subplot(121)
plt.title("Wavelet coefficients, exact")
plt.imshow(cExact, cmap=plt.cm.gray)
plt.subplot(122)
plt.title("Wavelet coefficients, noisy")
plt.imshow(cNoisy, cmap=plt.cm.gray)

"""
.. image:: PLOT2RST.current_figure

As we can see, the wavelet coefficient sub-bands become more "energetic" and
less when the image is contaminated with additive noise. This is a useful
property (separation of signal and noise in wavelet transform domain)
that is observable regardless of the chosen wavelet filter (though some
are more effective than others when proceeding with denoising).

We can denoise the image by applying thresholding functions
to each of the wavelet sub-bands, and then inverting the wavelet
transform. The `wavelet_filter` function will handle this all for us
automatically.

Now, we select a wavelet function, number of levels to visualize, and
level-specific coefficient thresholds. A full list of all usable wavelet
functions are available by calling wavelet_list() from skimage.filter,
"""

wavelet = "rbio6.8"
level = 3
t = [30, 30, 15]

"""
The optimal wavelet function is one that effectively separates important
features in an image from noise (or less important details). It is often
difficult to determine this a priori, though some general principles do apply.
For example, if edge information is the most important detail in an image,
then the Haar wavelet function is a good candidate.

The wavelets in the biorthogonal ('bior') and reverse-biorthogonal ('rbio')
adapt very well to symmetric features piece-wise smooth features, which are
often characteristic of "natural" images.

Now, we can perform wavelet denoising using our own defined thresholds by
calling the `wavelet_filter` function.
"""
denoised_custom = wavelet_filter(noisy, t, wavelet=wavelet, level=level)

f = plt.figure()
ax = f.add_subplot(111)
ax.set_axis_off()
description = "(%s, %s levels) \n t = %s" % (wavelet, level, t.__str__())
plt.title("Denoised " + description)
plt.imshow(denoised_custom, cmap=plt.cm.gray)

"""
.. image:: PLOT2RST.current_figure

The noise is significantly reduced in the result, though there are some
artifacts present. Effective reduction of noise and suppression of artifacts
can be achieved by careful selection of the wavelet function and threshold
values.

VisuShrink and BayesShrink
===========================
VisuShrink and BayesShrink are two commonly referenced wavelet-based image
denoising techniques. Each method has their own defined way to determine
wavelet coefficient threshold values for the wavelet filter. For more details,
see resources 3 and 4 below.

To perform wavelet denoising using the VisuShrink thresholds, we simply call
the `visu_shrink` function directly. The arguments that can be passed are the
same as those for `wavelet_filter` function, except that it does not accept
the coefficient threshold as an argument. Instead, the thresholds are
calculated them internally.
"""
denoised_visu = visu_shrink(noisy,  wavelet=wavelet, level=level)

f = plt.figure()
ax = f.add_subplot(111)
ax.set_axis_off()
description = "(%s, %s levels)" % (wavelet, level)
plt.title("Denoised, VisuShrink, " + description)
plt.imshow(denoised_visu, cmap=plt.cm.gray)

"""
.. image:: PLOT2RST.current_figure

VisuShrink applies a single calculated threshold value to all detail sub-bands
in the wavelet coefficient expansion.
The VisuShrink is often reported as giving very smooth-looking results, though
they may be too smooth in many instances.

BayesShrink can be performed by calling the `bayes_shrink` function directly,
just as in the case of VisuShrink.
"""
denoised_bayes = bayes_shrink(noisy,  wavelet=wavelet, level=level)

f = plt.figure()

ax = f.add_subplot(111)
ax.set_axis_off()
description = "(%s, %s levels)" % (wavelet, level)
plt.title("Denoised, BayesShrink, " + description)
plt.imshow(denoised_bayes, cmap=plt.cm.gray)

plt.show()

"""
.. image:: PLOT2RST.current_figure

BayesShrink is an adaptive thresholding method, which calculates a separate
coefficient threshold for each of the individual detail sub-bands. It adapts
well to various noise levels.

Additional Resources
====================

1. 'A really friendly guide to wavelets <http://math.ecnu.edu.cn/~qgu/friendintro.pdf>'

2. 'Image Denoising With Wavelets <https://www.ceremade.dauphine.fr/~peyre/numerical-tour/tours/denoisingwav_2_wavelet_2d/>'

3.   Donoho, David L., and Jain M. Johnstone. "Ideal spatial adaptation by
    wavelet shrinkage." Biometrika 81.3 (1994): 425-455.

4.  Chang, S. Grace, Bin Yu, and Martin Vetterli. "Adaptive wavelet
    thresholding for image denoising and compression." Image Processing, IEEE
    Transactions on 9.9 (2000): 1532-1546.
"""
