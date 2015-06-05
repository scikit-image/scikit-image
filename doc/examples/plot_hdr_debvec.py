"""
===========
HDR Image
===========

The hdr image is a combination of bracketed images into one (a series of images
taken with different exposure times)

In this example, we show the use of a image series to create a hdr image

The Debevec algorithm is published by:
Debevec and Malik, J. (1997). doi:10.1145/258734.258884
High dynamic imageing is nicely covered at 'Wikipedia
<https://en.wikipedia.org/wiki/High-dynamic-range_imaging>`

"""

import matplotlib.pyplot as plt

import numpy as np
from skimage.exposure import hdr
from skimage import data
from skimage.morphology import disk
from matplotlib.colors import LogNorm
from skimage.filters import rank
from skimage.color import rgb2gray

ims, exp = data.hdr_images()
exp = np.array(exp)


radiance_map = hdr.get_crf(ims, exp, depth=8)
plt.plot(radiance_map)
plt.show()
hdr_im = hdr.make_hdr(ims, exp, radiance_map, depth=8)

hdr_norm = np.zeros_like(hdr_im)
for ii in range(3):
    norm = np.max(np.nan_to_num(hdr_im[:, :, ii].flatten()))
    hdr_norm[:, :, ii] = hdr_im[:, :, ii] / norm

plt.imshow(hdr_norm)
plt.show()

I = 0.2125 * hdr_im[:, :, 0] + 0.7154 * \
    hdr_im[:, :, 1] + 0.0721 * hdr_im[:, :, 2]
# I = np.asanyarray(rgb2gray(hdr_im), dtype=np.float)


plt.imshow(I, 'gray', norm=LogNorm())
plt.show()


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
