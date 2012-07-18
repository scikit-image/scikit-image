"""
"""
print __doc__

import matplotlib.pyplot as plt
import numpy as np

from skimage.data import lena
from skimage.segmentation import km_segmentation
from skimage.util import img_as_float

img = img_as_float(lena()).copy("C")
segments = km_segmentation(img, ratio=2.0, n_segments=200)

print("number of segments: %d" % len(np.unique(segments)))

plt.subplot(131, title="original")
plt.imshow(img, interpolation='nearest')
plt.axis("off")

plt.subplot(132, title="superpixels")
# shuffle the labels for better visualization
plt.imshow(segments, interpolation='nearest', cmap=plt.cm.prism)
plt.axis("off")

plt.subplot(133, title="mean color")
colors = [np.bincount(segments.ravel(), img[:, :, c].ravel()) for c in
        xrange(img.shape[2])]
counts = np.bincount(segments.ravel())
colors = np.vstack(colors) / counts
plt.imshow(colors.T[segments], interpolation='nearest')
plt.axis("off")

plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)
plt.show()
