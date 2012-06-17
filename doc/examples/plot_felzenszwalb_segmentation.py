import matplotlib.pyplot as plt
import numpy as np

from skimage.data import lena
from skimage.segmentation import felzenszwalb_segmentation
from skimage.util import img_as_float

img = img_as_float(lena())
segments = felzenszwalb_segmentation(img, scale=1)
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
plt.show()
print("num segments: %d" % len(np.unique(segments)))
