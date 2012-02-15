"""
============================
Block views on images/arrays
============================

This example illustrates the use of `view_as_blocks` from
`skimage.util.shape`.  Block views can be incredibly useful when one
wants to perform local operations on non-overlapping image patches.

We use `lena` from `skimage.data` and virtually 'slice' it into square
blocks.  Then, on each block, we either pool the mean, the max or the
median value of that block. The results are displayed altogether, along
with a 'classic' `bicubic` rescaling of the original `lena` image.
"""

import numpy as np
from scipy.misc import imresize
from matplotlib import pyplot as plt
import matplotlib.cm as cm

from skimage import data
from skimage import color
from skimage.util.shape import view_as_blocks


# -- get `lena` from skimage.data in grayscale
l = color.rgb2gray(data.lena()) / 255.

# -- size of blocks
block_shape = (4, 4)

# -- see `lena` as a matrix of blocks (of shape
#    `block_shape`)
view = view_as_blocks(l, block_shape)

# -- collapse the last two dimensions in one
flatten_view = view.reshape(view.shape[0], view.shape[1], -1)

# -- resampling `lena` by taking either the `mean`,
#    the `max` or the `median` value of each blocks.
mean_view = np.mean(flatten_view, axis=2)
max_view = np.max(flatten_view, axis=2)
median_view = np.median(flatten_view, axis=2)

# -- display resampled images
plt.figure(figsize=(10, 10))

plt.subplot(221)
plt.title("Original rescaled\n in bicubic mode")
l_resized = imresize(l, view.shape[:2], interp='bicubic')
plt.imshow(l_resized, cmap=cm.Greys_r)

plt.subplot(222)
plt.title("Block view with\n local mean pooling")
plt.imshow(mean_view, cmap=cm.Greys_r)

plt.subplot(223)
plt.title("Block view with\n local max pooling")
plt.imshow(max_view, cmap=cm.Greys_r)

plt.subplot(224)
plt.title("Block view with\n local median pooling")
plt.imshow(median_view, cmap=cm.Greys_r)

plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.show()
