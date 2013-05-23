"""
==========================================
Find the intersection of two segmentations
==========================================

When segmenting an image, you may want to combine multiple alternative
segmentations. The `skimage.segmentation.join_segmentations` function
computes the join of two segmentations, in which a pixel is placed in
the same segment if and only if it is in the same segment in _both_
segmentations.

"""
import numpy as np
from scipy import ndimage as nd
import matplotlib.pyplot as plt
import matplotlib as mpl

from skimage.filter import sobel
from skimage.segmentation import slic, join_segmentations
from skimage.morphology import watershed
from skimage import data


coins = data.coins()

# make segmentation using edge-detection and watershed
edges = sobel(coins)
markers = np.zeros_like(coins)
foreground, background = 1, 2
markers[coins < 30] = background
markers[coins > 150] = foreground

ws = watershed(edges, markers)
seg1 = nd.label(ws == foreground)[0]

# make segmentation using SLIC superpixels

# make the RGB equivalent of `coins`
coins_colour = np.tile(coins[..., np.newaxis], (1, 1, 3))
seg2 = slic(coins_colour, n_segments=30, max_iter=160, sigma=1, ratio=9,
            convert2lab=False)

# combine the two
segj = join_segmentations(seg1, seg2)

### Display the result ###

# make a random colormap for a set number of values
def random_cmap(im):
    np.random.seed(9)
    cmap_array = np.concatenate(
        (np.zeros((1, 3)), np.random.rand(np.ceil(im.max()), 3)))
    return mpl.colors.ListedColormap(cmap_array)

# show the segmentations
fig, axes = plt.subplots(ncols=4, figsize=(9, 2.5))
axes[0].imshow(coins, cmap=plt.cm.gray, interpolation='nearest')
axes[0].set_title('Image')
axes[1].imshow(seg1, cmap=random_cmap(seg1), interpolation='nearest')
axes[1].set_title('Sobel+Watershed')
axes[2].imshow(seg2, cmap=random_cmap(seg2), interpolation='nearest')
axes[2].set_title('SLIC superpixels')
axes[3].imshow(segj, cmap=random_cmap(segj), interpolation='nearest')
axes[3].set_title('Join')

for ax in axes:
    ax.axis('off')
plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
plt.show()
