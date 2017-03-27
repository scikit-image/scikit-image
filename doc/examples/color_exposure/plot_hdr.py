"""
===========
HDR Image
===========

A HDR (High Dynamic Range) image is a combination of bracketed images (varying
exposures) into one.

In this example, we show the use of a series of images at different exposures
to create a HDR image.

References
----------

.. [1] Debevec and Malik, J. (1997). DOI:10.1145/258734.258884

.. [2] https://en.wikipedia.org/wiki/High-dynamic-range_imaging
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.exposure import adjust_gamma, hdr
from skimage import data

# Get example images
ims, exp = data.hdr_images()
exp = np.array(exp)

# Get radiance map (how the radiance maps to the counts for each channel)
radiance_map = hdr.get_crf(ims, exp, depth=8, l=100)

# Show radiance map
plt.title('Camera response function')
plt.xlabel('Counts')
plt.ylabel('Radiance')
colors = ['r', 'g', 'b']
for ii in range(radiance_map.shape[1]):
    plt.plot(radiance_map[:, ii], colors[ii])
plt.legend(['Red', 'Green', 'Blue'], loc='best')
plt.show()

# Make the HDR image
hdr_im = hdr.make_hdr(ims, exp, radiance_map, depth=8)

# Normalise the hdr image
hdr_norm = np.zeros_like(hdr_im)
norm = np.max(np.nan_to_num(hdr_im.flatten()))
for ii in range(3):
    hdr_norm[:, :, ii] = hdr_im[:, :, ii] / norm


fig, axes = plt.subplots(nrows=1, ncols=2)
# Show hdr image. This is going to be dark due to the range in the image
axes[0].imshow(hdr_norm)
axes[0].set_title("HDR image")
axes[0].set_axis_off()
# Show gamma adjusted hdr image.
axes[1].imshow(adjust_gamma(hdr_norm, gamma=0.25))
axes[1].set_title("HDR image gamma adjusted")
axes[1].set_axis_off()
plt.show()

# Below follows a commented out example for saving the image as a hdr image
# importable by other processing software
# from skimage.io import imsave
# imsave(fname, hdr_norm.astype(np.float32), plugin='tifffile')


# Plotting a histogram equalised  hdr image.
# from skimage.morphology import disk
# from matplotlib.colors import LogNorm
# from skimage.filters import rank
# from skimage.color import rgb2gray
# hdr_hist = np.zeros_like(hdr_norm)
# for ii in range(3):
#     hdr_hist[:, :, ii] = hdr_norm / (hdr_norm + 1)
# tone_mapped = np.zeros_like(hdr_im)
# selem = disk(10)
# eq = rank.equalize(I, selem=selem)
# for i in range(3):
#     norm = np.max(np.nan_to_num(hdr_im[:, :, ii].flatten()))
#     tone_mapped[:, :, i] = hdr_im[:, :, i] * eq / (I * norm)
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
# axes[0].imshow(I, 'gray', norm=LogNorm())
# axes[1].imshow(tone_mapped)
# plt.show()
