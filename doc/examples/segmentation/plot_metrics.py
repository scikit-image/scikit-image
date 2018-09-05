"""
====================================================
Comparison of segments
====================================================

In this example we will:
 * compute adapted rand error as defined by the SNEMI3D contest
 * find the variation of information between two segments
 * find the split variation of information between two segments

"""

import matplotlib.pyplot as plt
import numpy as np

from skimage.data import coffee
from skimage.measure import (compare_adapted_rand_error,
                             compare_split_variation_of_information,
                             compare_variation_of_information)
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

img = coffee()

segments_slic = slic(img, n_segments=2, compactness=10, sigma=1)
gradient = sobel(rgb2gray(img))
segments_watershed = watershed(gradient, markers=2, compactness=0.001)

print('Adapated rand error: {}'.format(compare_adapted_rand_error(segments_slic, segments_watershed)))
print('Split variation of information: {}'.format(compare_split_variation_of_information(segments_watershed[1], segments_slic[1])))
print('Variation of information: {}'.format(compare_variation_of_information(segments_watershed[1], segments_slic[1], weights=np.array([1, 2]))))

fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0].imshow(mark_boundaries(img, segments_slic))
ax[0].set_title('SLIC')
ax[1].imshow(mark_boundaries(img, segments_watershed))
ax[1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()
