"""
==================
Saliency Detection
==================

Detect salient regions by assigning a saliency score to them which is
measured by the entropy of a region's descriptor (to find locally complex regions)
weighed by the difference over multiple scales to select features
that are globally discriminative.

"""

from skimage import data
from skimage.feature import saliency_kadir_brady
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

image = data.astronaut()
image_gray = rgb2gray(image)

regions = saliency_kadir_brady(image_gray, min_scale=5, max_scale=13,
                               saliency_threshold=0.6, clustering_threshold=2)

fig,ax = plt.subplots()
ax.axis('off')

for y,x,r in regions:
    c = plt.Circle((x,y), r, color='red', linewidth=2, fill=False)
    ax.add_patch(c)

ax.imshow(image, interpolation='nearest')
plt.show()
