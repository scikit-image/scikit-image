from __future__ import division
"""
========================
Sliding window histogram
========================

This example extracts a single coin from the `skimage.data.coins` image and
generates a histogram of its greyscale values.

It then computes a sliding window histogram of the complete image using
`skimage.filter.rank.windowed_histogram`. The local histogram for the region
surrounding each pixel in the image is compared to that of the single coin,
with a similarity measure being computed and displayed.

To demonstrate the rotational invariance of the technique, the same
test is performed on a version of the coins image rotated by 45 degrees.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from skimage import data
from skimage.util import img_as_ubyte
from skimage.morphology import disk
from skimage.filter import rank
from skimage import transform


matplotlib.rcParams['font.size'] = 9


def windowed_histogram_similarity(image, selem, reference_hist, n_bins):
    # Compute normalized windowed histogram feature vector for each pixel
    px_histograms = rank.windowed_histogram(image, selem, n_bins=n_bins)

    # Reshape coin histogram to (1,1,N) for broadcast when we want to use it in
    # arithmetic operations with the windowed histograms fro the image
    reference_hist = reference_hist.reshape((1,1) + reference_hist.shape)

    # Compute Chi squared distance metric: sum((X-Y)^2 / (X+Y));
    # a measure of distance between histograms
    X = px_histograms
    Y = reference_hist
    num = (X-Y)*(X-Y)
    denom = X+Y
    frac = num / denom
    frac[denom==0] = 0
    chi_sqr = np.sum(frac, axis=2) * 0.5

    # Generate a similarity measure. It needs to be low when distance is high.
    # and high when distance is low; taking the reciprocal will do this.
    # Chi squared will always be >= 0, add small value to prevent divide by 0.
    similarity = 1 / (chi_sqr + 1.0e-4)

    return similarity


# Load the `skimage.data.coins` image
img = img_as_ubyte(data.coins())

# Quantize to 16 levels of grayscale; this way the output image will have a
# 16-dimensional feature vector per pixel
quantized_img = img//16

# Select the coin from the 4th column, second row.
# Co-ordinate ordering: [x1,y1,x2,y2]
coin_coords = [184,100,228,148]   # 44 x 44 region
coin = quantized_img[coin_coords[1]:coin_coords[3], coin_coords[0]:coin_coords[2]]

# Compute coin histogram and normalize
coin_hist, _ = np.histogram(coin.flatten(), bins=16, range=(0,16))
coin_hist = coin_hist.astype(float) / np.sum(coin_hist)


# Compute a disk shaped mask that will define the shape of our sliding window
# Example coin is ~44px across, so make a disk 61px wide (2*rad+1) to be big
# enough for other coins too.
selem = disk(30)


# Compute the similarity across the complete image
similarity = windowed_histogram_similarity(quantized_img, selem, coin_hist,
                                           coin_hist.shape[0])

# Now try a rotated image
rotated_img = img_as_ubyte(transform.rotate(img, 45.0, resize=True))
# Quantize to 16 levels as before
quantized_rotated_image = rotated_img//16
# Similarity on rotated image
rotated_similarity = windowed_histogram_similarity(quantized_rotated_image,
                                                   selem, coin_hist,
                                                   coin_hist.shape[0])



# Plot it all
fig, axes = plt.subplots(nrows=5, figsize=(6, 18))
ax0, ax1, ax2, ax3, ax4 = axes

ax0.imshow(img, cmap='gray')
ax0.set_title('Original image')
ax0.axis('off')

ax1.imshow(quantized_img, cmap='gray')
ax1.set_title('Quantized image')
ax1.axis('off')

ax2.imshow(coin, cmap='gray')
ax2.set_title('Coin from 2nd row, 4th column')
ax2.axis('off')

ax3.imshow(img, cmap='gray')
# While jet is not a great colormap, it makes the high similarity areas
# stand out
ax3.imshow(similarity, cmap='jet', alpha=0.5)
ax3.set_title('Original image with overlayed similarity')
ax3.axis('off')

ax4.imshow(rotated_img, cmap='gray')
ax4.imshow(rotated_similarity, cmap='jet', alpha=0.5)
ax4.set_title('Rotated image with overlayed similarity')
ax4.axis('off')

plt.show()
