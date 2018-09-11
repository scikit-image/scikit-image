"""
====================================================
Comparison of segments
====================================================

In this example we will:
 * compute adapted rand error as defined by the SNEMI3D contest
 * find the variation of information between two segments
 * find the split variation of information between two segments

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes

from skimage import data
from skimage.measure import (compare_adapted_rand_error,
                             compare_split_variation_of_information,
                             compare_variation_of_information)
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.util import img_as_float, regular_grid
from skimage.feature import canny
from skimage.morphology import remove_small_objects
from skimage.segmentation import (morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient, 
                                  watershed, 
                                  mark_boundaries)

image = data.coins()

#Region based
elevation_map = sobel(image)
markers = np.zeros_like(image)
markers[image < 30] = 1
markers[image > 150] = 2
im_true = watershed(elevation_map, markers)
im_true = binary_fill_holes(im_true - 1)

# Compact watershed
edges = sobel(image)
grid = regular_grid(image.shape, n_points=468)
seeds = np.zeros(image.shape, dtype=int)
seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape) + 1
im_test1 = watershed(edges, seeds, compactness=0.00001)

# Edge based
edges = canny(image)
fill_coins = binary_fill_holes(edges)
im_test2 = remove_small_objects(fill_coins, 21)

# Morphological GAC
image = img_as_float(image)
gimage = inverse_gaussian_gradient(image)
init_ls = np.zeros(image.shape, dtype=np.int8)
init_ls[10:-10, 10:-10] = 1
im_test3 = morphological_geodesic_active_contour(gimage, 100, init_ls,
                                           smoothing=1, balloon=-1,
                                           threshold=0.69)

precision_list = []
recall_list = []
split_list = []
merge_list = []
for im_test in [im_test1, im_test2, im_test3]:
    error, precision, recall = compare_adapted_rand_error(im_true, im_test)
    splits, merges = compare_split_variation_of_information(im_true, im_test)
    split_list.append(splits)
    merge_list.append(merges)
    precision_list.append(precision)
    recall_list.append(recall)
    print('Adapated Rand error: %f' % error)
    print('Adapted Rand precision: %f' % precision)
    print('Adapted Rand recall: %f' % recall)
    print('False Splits: %f' % splits)
    print('False Merges: %f' % merges)

fig, axes = plt.subplots(2, 3, figsize=(10, 10))
ax = axes.ravel()

ax[0] = plt.subplot(2, 3, 1)
ax[0].scatter(merge_list, split_list)
for i, txt in enumerate(['Compact', 'Edge', 'GAC']):
    ax[0].annotate(txt, (merge_list[i], split_list[i]))
ax[0].set_xlabel('False Merges')
ax[0].set_ylabel('False Splits')
ax[0].set_title("Split Variation of Information")

ax[1] = plt.subplot(2, 3, 2)
ax[1].scatter(precision_list, recall_list)
for i, txt in enumerate(['Compact', 'Edge', 'GAC']):
    ax[1].annotate(txt, (precision_list[i], recall_list[i]))
ax[1].set_xlabel('Adapted Rand precision')
ax[1].set_ylabel('Adapted Rand recall')
ax[1].set_title("Precision vs. Recall")

ax[2].imshow(mark_boundaries(image, im_true))
ax[2].set_title('True Segmentation')
ax[2].set_axis_off()

ax[3].imshow(mark_boundaries(image, im_test1))
ax[3].set_title('Compact Watershed')
ax[3].set_axis_off()

ax[4].imshow(mark_boundaries(image, im_test2))
ax[4].set_title('Edge Detection')
ax[4].set_axis_off()

ax[5].imshow(mark_boundaries(image, im_test3))
ax[5].set_title('Morphological GAC')
ax[5].set_axis_off()

plt.tight_layout()
plt.show()
