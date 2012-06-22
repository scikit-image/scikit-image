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
segments = quickshift(img, kernel_size=5, max_dist=20)

print("number of segments: %d" % len(np.unique(segments)))

fig, (ax_org, ax_sp, ax_mean) = plt.subplots(1, 3)
ax_org.set_title("original")
ax_org.imshow(img, interpolation='nearest')
ax_org.axis("off")

ax_sp.set_title("superpixels")
ax_sp.imshow(segments, interpolation='nearest', cmap=plt.cm.prism)
ax_sp.axis("off")

colors = [np.bincount(segments.ravel(), img[:, :, c].ravel()) for c in
        xrange(img.shape[2])]
counts = np.bincount(segments.ravel())
colors = np.vstack(colors) / counts
ax_mean.set_title("mean color")
ax_mean.imshow(colors.T[segments], interpolation='nearest')
ax_mean.axis("off")
fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)
plt.show()
