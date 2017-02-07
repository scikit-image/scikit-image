"""
===========
HDR Image
===========

The hdr image is a combination of bracketed images into one (a series of images
taken with different exposure times)

In this example, we show the use of a image series to create a hdr image

The Debevec algorithm is published by:
Debevec and Malik, J. (1997). DOI:10.1145/258734.258884
High dynamic imageing is nicely covered at 'Wikipedia
<https://en.wikipedia.org/wiki/High-dynamic-range_imaging>`

"""

import matplotlib.pyplot as plt

import numpy as np
from skimage.exposure import adjust_gamma, hdr
from skimage import data
from skimage.morphology import disk
from matplotlib.colors import LogNorm
from skimage.filters import rank
from skimage.color import rgb2gray
from skimage.io import imsave
import skimage

# Get example images
ims, exp = data.hdr_images()
exp = np.array(exp)

# Get radiance map (how the radiance maps to the counts for each channel
radiance_map = hdr.get_crf(ims, exp, depth=8, l=100)

# Show radiance map
plt.plot(radiance_map)
plt.show()

# Make the HDR image
hdr_im = hdr.make_hdr(ims, exp, radiance_map, depth=8)


# Normalise the hdr image
hdr_norm = np.zeros_like(hdr_im)
norm = np.max(np.nan_to_num(hdr_im.flatten()))
for ii in range(3):
    hdr_norm[:, :, ii] = hdr_im[:, :, ii] / norm


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
# Show hdr image. This is going to be dark due to the range in the image
axes[0].imshow(hdr_norm)
# Show gamma adjusted hdr image.
axes[1].imshow(adjust_gamma(hdr_norm, gamma=0.25))
#
# print(np.max(np.nan_to_num(hdr_hist)))
# axes[2].imshow(hdr_im / (hdr_im + 1))
plt.show()

## Below follows a commented out example for saving the image as a hdr image 
## importable by other processing software
## We only have on working format, tiff, as currently the freeimage plugin 
## does not export to .hdr  and .exr files correctly
# imsave(fname, hdr_norm.astype(np.float32), plugin='tifffile')


## Plotting a histogram equalized  hdr image.
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
