"""
======================
Wavelet Denoising
======================

This example shows the results of a simple wavelet denoising filter.

An n-level discrete wavelet transform decomposes a digital image into 1 + 3*n
coefficient matrices. The first is called the approximation subband, which
captures the mean (and other low-frequency information) of the original image,
and is essentially a down-sampled representation of it.

The remaining coefficient matrices are called the detail subbands. Thresholding
functions are applied to these matrices. The wavelet transform is then
reversed, to give the denoised result.
Thresholding can either be soft or hard[1].

The main variation between the wavelet denoising methods that are out there in
the literature are in how the thresholding values for each detail subband is
determined (whether it is uniform across all detail subbands, or individual
for each, etc).

The `wavelet_filter` function allows a user to specify the detail thresholds
either uniformly, or by level, or by individual subband. This way, this
function can be used to implement the majority of wavelet denoising methods
out there (BayesShrink. SureShrink. etc) by writing short wrapping functions
which pre-compute the thresholds accordingly.

Arguments to wavelet_filter are the image to be
denoised, threshold value(s), and all arguments which could be passed to
pywt.wavedec2. Note that in wavelet_filter, the wavelet function to use is an
optional arguement, and defaults to haar.

[1] http://en.wikipedia.org/wiki/Discrete_wavelet_transform
[2] http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/xlghtmlnode93.html
"""
import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage.filter import wavelet_filter


# Load an example image & project to grayscale

img = data.lena().mean(axis=2)

# Add some noise
m, n, = img.shape
noisy = img + 15 * np.random.randn(m, n)

# Select wavelet, number of levels, and level-specific thresholds
wavelet = "bior5.5"
level = 4
t = [30, 30, 15, 15]

# Perform wavelet denoising
denoised = wavelet_filter(noisy, t, wavelet=wavelet, level=level)

f1 = plt.figure()
ax = f1.add_subplot(111)
ax.set_axis_off()
ax.autoscale_view(True, True, True)
plt.title("Exact")
plt.imshow(img, cmap=plt.cm.gray)

f2 = plt.figure()
ax = f2.add_subplot(111)
ax.set_axis_off()
ax.autoscale_view(True, True, True)
plt.title("Noisy")
plt.imshow(noisy, cmap=plt.cm.gray)

f3 = plt.figure()
ax = f3.add_subplot(111)
ax.set_axis_off()
ax.autoscale_view(True, True, True)
description = "(%s, %s levels)" % (wavelet, level)
plt.title("Denoised " + description)
plt.imshow(denoised, cmap=plt.cm.gray)
plt.show()
