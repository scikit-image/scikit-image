"""
===================
Canny edge detector
===================

The Canny filter is a multi-stage edge detector. It uses a filter based on the
derivative of a Gaussian in order to compute the intensity of the gradients.The
Gaussian reduces the effect of noise present in the image. Then, potential
edges are thinned down to 1-pixel curves by removing non-maximum pixels of the
gradient magnitude. Finally, edge pixels are kept or removed using hysteresis
thresholding on the gradient magnitude.

The Canny has three adjustable parameters: the width of the Gaussian (the
noisier the image, the greater the width), and the low and high threshold for
the hysteresis thresholding.

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import feature
from skimage.filters import threshold_otsu


# Generate noisy image of a square
im = np.zeros((128, 128))
im[32:-32, 32:-32] = 1

im = ndi.rotate(im, 15, mode='constant')
im = ndi.gaussian_filter(im, 4)
im += 0.2 * np.random.random(im.shape)

# Generate lower SNR noisy image of a square
im2 = np.zeros((128, 128))
im2[32:-32, 32:-32] = 1

im2 = ndi.rotate(im2, 15, mode='constant')
im2 = ndi.gaussian_filter(im2, 4)
im2 += 0.4 * np.random.randn(*(im2.shape))

# Compute the Canny filter for two values of sigma
edges1_1 = feature.canny(im)
edges1_2 = feature.canny(im, sigma=3)
edges1_3 = feature.canny(im, use_quantiles=True, sigma=3)
edges1_4 = feature.canny(im, sigma=3, high_threshold=threshold_otsu,
                         low_threshold=threshold_otsu)

edges2_1 = feature.canny(im2)
edges2_2 = feature.canny(im2, sigma=3)
edges2_3 = feature.canny(im2, sigma=3, use_quantiles=True)
edges2_4 = feature.canny(im2, sigma=3, low_threshold=threshold_otsu,
                         high_threshold=threshold_otsu)

# display results
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6),
                                    sharex=True, sharey=True)

axes[0, 0].imshow(im, cmap=plt.cm.gray)
axes[0, 0].set_title('Low noise image')

axes[0, 1].imshow(edges1_1, cmap=plt.cm.gray)
axes[0, 1].set_title('Canny filter, $\sigma=1$')

axes[0, 2].imshow(edges1_2, cmap=plt.cm.gray)
axes[0, 2].set_title('Canny filter, $\sigma=3$')

axes[0, 3].imshow(edges1_3, cmap=plt.cm.gray)
axes[0, 3].set_title('Canny filter, $\sigma=3$\nusing quantiles')

axes[0, 4].imshow(edges1_4,  cmap=plt.cm.gray)
axes[0, 4].set_title('Canny filter, $\sigma=3$\nOtsu threshold (callable)')

axes[1, 0].imshow(im2, cmap=plt.cm.gray)
axes[1, 0].set_title('High noise image')

axes[1, 1].imshow(edges2_1, cmap=plt.cm.gray)
axes[1, 1].set_title('Canny filter, $\sigma=1$\ndefault threshold')

axes[1, 2].imshow(edges2_2, cmap=plt.cm.gray)
axes[1, 2].set_title('Canny filter, $\sigma=3$\ndefault threshold')

axes[1, 3].imshow(edges2_3, cmap=plt.cm.gray)
axes[1, 3].set_title('Canny filter, $\sigma=3$\nquantiles')

axes[1, 4].imshow(edges2_4,  cmap=plt.cm.gray)
axes[1, 4].set_title('Canny filter, $\sigma=3$\nOtsu threshold (callable)')

for ax in axes.ravel():
    ax.axis('off')

fig.tight_layout()

plt.show()
