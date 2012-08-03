"""
=============================
Quickshift image segmentation
=============================

Quickshift is a relatively recent 2d image segmentation algorithm, based on an
approximation of kernelized mean-shift. Therefore it belongs to the family
of local mode-seeking algorithms and is applied to the color+coordinate space,
see [1]_ It is often used to extract "superpixels", small homogeneous image
regions, which build the basis for further processing.

One of the benefits of quickshift is that it actually computes a
hierarchical segmentation on multiple scales simultaneously.

Quickshift has two parameters, one controlling the scale of the local
density approximation, the other selecting a level in the hierarchical
segmentation that is produced.

.. [1] Quick shift and kernel methods for mode seeking, Vedaldi, A. and Soatto, S.
       European Conference on Computer Vision, 2008
"""
print __doc__

import matplotlib.pyplot as plt
import numpy as np

from skimage.data import lena
from skimage.segmentation import quickshift, visualize_boundaries
from skimage.util import img_as_float
from skimage.color import rgb2lab

img = img_as_float(lena())[::2, ::2, :].copy("C")
segments = quickshift(rgb2lab(img), kernel_size=5, max_dist=20)
segments_rgb = quickshift(img, kernel_size=5, max_dist=20)

print("number of segments: %d" % len(np.unique(segments)))
boundaries = visualize_boundaries(img, segments)
boundaries_rgb = visualize_boundaries(img, segments_rgb)
plt.imshow(boundaries)
plt.figure()
plt.imshow(boundaries_rgb)
plt.axis("off")
plt.show()
