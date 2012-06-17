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
from skimage.segmentation import quickshift
from skimage.util import img_as_float


img = img_as_float(lena())[::2, ::2, :].copy("C")
segments = quickshift(img, sigma=5, tau=20)
segments = np.unique(segments, return_inverse=True)[1].reshape(img.shape[:2])

plt.subplot(131, title="original")
plt.imshow(img, interpolation='nearest')

plt.subplot(132, title="superpixels")
# shuffle the labels for better visualization
permuted_labels = np.random.permutation(segments.max() + 1)
plt.imshow(permuted_labels[segments], interpolation='nearest')

plt.subplot(133, title="mean color")
colors = [np.bincount(segments.ravel(), img[:, :, c].ravel()) for c in
        xrange(img.shape[2])]
counts = np.bincount(segments.ravel())
colors = np.vstack(colors) / counts
plt.imshow(colors.T[segments], interpolation='nearest')
print("number of segments: %d" % len(np.unique(segments)))
plt.show()
