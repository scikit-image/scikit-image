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

There are two adjustable parameters: the window_size, which controls the
size of the neighbourhood considered in computing the statistics, and eta,
which controls the strength of the smoothing. Larger eta's approximately
correspond to stronger smoothing.

==========
References
==========

.. [1] http://research.microsoft.com/en-us/um/people/kahe/eccv10/


"""

from skimage.data import immunohistochemistry
from skimage.restoration import guided_filter
import numpy as np
import matplotlib.pyplot as plt


# Create a gray rectangle on a black background and add some large amplitude
# gaussian noise.
gray_square = np.zeros((400, 400))
gray_square[100:300, 100:300] = 0.5
noisy_square = gray_square + np.random.normal(scale=0.4, size=gray_square.shape)
noisy_square = np.clip(noisy_square, 0, 1)

# Filter the noisy gray square with itself at lower and higher eta, with the
# same window radius for each.
guided_low = guided_filter(noisy_square, 0.1, 5)
guided_high = guided_filter(noisy_square, 0.5, 5)

# Plot the results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

ax1.imshow(noisy_square, cmap='gray', vmin=0, vmax=1)
ax1.axis('off')
ax1.set_title('Original')

ax2.imshow(guided_low, cmap='gray', vmin=0, vmax=1)
ax2.axis('off')
ax2.set_title('Eta = 0.1')

ax3.imshow(guided_high, cmap='gray', vmin=0, vmax=1)
ax3.axis('off')
ax3.set_title('Eta = 0.5')

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()

# Filter the noisy gray square with the original image at lower and higher eta,
# with the same window radius for each.
guided_low = guided_filter(noisy_square, 0.1, 5, guide=gray_square)
guided_high = guided_filter(noisy_square, 0.5, 5, guide=gray_square)

# Plot the results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

ax1.imshow(noisy_square, cmap='gray', vmin=0, vmax=1)
ax1.axis('off')
ax1.set_title('Original')

ax2.imshow(guided_low, cmap='gray', vmin=0, vmax=1)
ax2.axis('off')
ax2.set_title('Eta = 0.1')

ax3.imshow(guided_high, cmap='gray', vmin=0, vmax=1)
ax3.axis('off')
ax3.set_title('Eta = 0.5')

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()


# Load a colour image and add gaussian noise idependently to each of the
# channels of a colour image.
ihc = immunohistochemistry()/255.0
ihc_noisy = ihc + np.random.normal(scale=0.3, size=ihc.shape)
ihc_noisy = np.clip(ihc_noisy, 0, 1)
ihc_noisy_gray = ihc_noisy.mean(axis=2)

guided_low = guided_filter(ihc_noisy, 0.02, 5, guide=ihc_noisy_gray)
guided_high = guided_filter(ihc_noisy, 0.1, 5, guide=ihc_noisy_gray)

# Plot the results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

ax1.imshow(ihc_noisy)
ax1.axis('off')
ax1.set_title('Original with noise')

ax2.imshow(guided_low)
ax2.axis('off')
ax2.set_title('Eta = 0.02')

ax3.imshow(guided_high)
ax3.axis('off')
ax3.set_title('Eta = 0.1')

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()
