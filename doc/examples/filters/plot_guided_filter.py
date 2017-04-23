"""
=============
Guided filter
=============

The guided filter is a fast, non-approximate edge preserving filter. It uses
the local pixel statistics of an input image and a guide image to solve a
linear regression problem to compute the smoothed image.

The authors make the following comment: "Try the guided filter in any situation
when the bilateral filter works well. The guided filter is much faster and
sometimes (though not always) works even better." [1]_

There are two adjustable parameters: the `window_size`, which controls the
size of the neighbourhood considered in computing the statistics, and `eta`,
which controls the strength of the smoothing. Larger `eta`'s approximately
correspond to stronger smoothing.

.. [1] Guided Image Filtering, Kaiming He, Jian Sun, and Xiaoou Tang,
       http://kaiminghe.com/eccv10/index.html, 2010,
       DOI: 10.1007/978-3-642-15549-9_1
"""

import numpy as np
from skimage.data import immunohistochemistry
from skimage.restoration import guided_filter
import matplotlib.pyplot as plt


# Create a gray rectangle on a black background and add some gaussian noise.
square_gray = np.zeros((400, 400))
square_gray[100:300, 100:300] = 0.5
square_noisy = square_gray + np.random.normal(
    scale=0.4, size=square_gray.shape)
square_noisy = np.clip(square_noisy, 0, 1)

# Filter the noisy gray square with itself at lower and higher eta,
# with the same window radius for each.
guided_low = guided_filter(square_noisy, 0.1, 5)
guided_high = guided_filter(square_noisy, 0.5, 5)

# Visualize the results
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))
ax = axes.flatten()

ax[0].imshow(square_noisy, cmap='gray', vmin=0, vmax=1)
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(guided_low, cmap='gray', vmin=0, vmax=1)
ax[1].axis('off')
ax[1].set_title('Eta = 0.1')

ax[2].imshow(guided_high, cmap='gray', vmin=0, vmax=1)
ax[2].axis('off')
ax[2].set_title('Eta = 0.5')

fig.tight_layout()
plt.show()

# Filter the noisy gray square with the original image at lower and higher eta,
# with the same window radius for each.
guided_low = guided_filter(square_noisy, 0.1, 5, guide=square_gray)
guided_high = guided_filter(square_noisy, 0.5, 5, guide=square_gray)

# Visualize the results
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))
ax = axes.flatten()

ax[0].imshow(square_noisy, cmap='gray', vmin=0, vmax=1)
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(guided_low, cmap='gray', vmin=0, vmax=1)
ax[1].axis('off')
ax[1].set_title('Eta = 0.1')

ax[2].imshow(guided_high, cmap='gray', vmin=0, vmax=1)
ax[2].axis('off')
ax[2].set_title('Eta = 0.5')

fig.tight_layout()
plt.show()

# Load a color image and add gaussian noise independently
# to each of the channels of a color image.
ihc = immunohistochemistry() / 255.0
ihc_noisy = ihc + np.random.normal(scale=0.3, size=ihc.shape)
ihc_noisy = np.clip(ihc_noisy, 0, 1)
ihc_noisy_gray = ihc_noisy.mean(axis=2)

guided_low = guided_filter(ihc_noisy, 0.02, 5, guide=ihc_noisy_gray)
guided_high = guided_filter(ihc_noisy, 0.1, 5, guide=ihc_noisy_gray)

# Visualize the results
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))
ax = axes.flatten()

ax[0].imshow(ihc_noisy)
ax[0].axis('off')
ax[0].set_title('Original with noise')

ax[1].imshow(guided_low)
ax[1].axis('off')
ax[1].set_title('Eta = 0.02')

ax[2].imshow(guided_high)
ax[2].axis('off')
ax[2].set_title('Eta = 0.1')

fig.tight_layout()
plt.show()
