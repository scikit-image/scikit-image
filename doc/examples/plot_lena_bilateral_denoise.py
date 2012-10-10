
"""
====================================================
Denoising the picture of Lena using bilateral filter
====================================================

In this example, we denoise a noisy version of the picture of Lena
using an approximation of a bilateral filter.
The pixels used to compute a local mean respect these conditions:
- be close to the central pixel, i.e. belong to the given structuring element.
- have a similar gray level, similarity is fixed by an interval [-s0,+s1] centered on the central pixel gray level.

The filter used is an approximation of a classical bilateral filter in the sens that kernel are usually gaussian
both in spatial and spectral dimensions.
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage.rank import bilateral_mean
from skimage.morphology import disk

l = img_as_ubyte(color.rgb2gray(data.lena()))
l = l[230:290, 220:320]

noisy = l + 0.4 * l.std() * np.random.random(l.shape)

selem = disk(30)
bilateral_denoised = bilateral_mean(noisy.astype(np.uint8), selem=selem,s0=10,s1=10)

plt.figure(figsize=(8, 2))

plt.subplot(131)
plt.imshow(noisy, cmap=plt.cm.gray, vmin=40, vmax=220)
plt.axis('off')
plt.title('noisy', fontsize=20)
plt.subplot(132)
plt.imshow(bilateral_denoised, cmap=plt.cm.gray, vmin=40, vmax=220)
plt.axis('off')
plt.title('bilateral denoising', fontsize=20)

selem = disk(30)
bilateral_denoised = bilateral_mean(noisy.astype(np.uint8), selem=selem,s0=40,s1=40)
plt.subplot(133)
plt.imshow(bilateral_denoised, cmap=plt.cm.gray, vmin=40, vmax=220)
plt.axis('off')
plt.title('(more) bilateral denoising', fontsize=20)

plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0, left=0,right=1)
plt.show()
