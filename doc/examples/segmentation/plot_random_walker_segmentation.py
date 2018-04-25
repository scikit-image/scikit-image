"""
==========================
Random walker segmentation
==========================

The random walker algorithm [1]_  determines the segmentation of an image from
a set of markers labeling several phases (2 or more). An anisotropic diffusion
equation is solved with tracers initiated at the markers' position. The local
diffusivity coefficient is greater if neighboring pixels have similar values,
so that diffusion is difficult across high gradients. The label of each unknown
pixel is attributed to the label of the known marker that has the highest
probability to be reached first during this diffusion process.

In this example, two phases are clearly visible, but the data are too
noisy to perform the segmentation from the histogram only. We determine
markers of the two phases from the extreme tails of the histogram of gray
values, and use the random walker for the segmentation.

.. [1] *Random walks for image segmentation*, Leo Grady, IEEE Trans. Pattern
       Anal. Mach. Intell. 2006 Nov; 28(11):1768-83 DOI:10.1109/TPAMI.2006.233

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import skimage

# Generate noisy synthetic data
data = skimage.img_as_float(binary_blobs(length=128, seed=1))
sigma = 0.35
data += np.random.normal(loc=0, scale=sigma, size=data.shape)
data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                         out_range=(-1, 1))

# The range of the binary image spans over (-1, 1).
# We choose the hottest and the coldest pixels as markers.
markers = np.zeros(data.shape, dtype=np.uint)
markers[data < -0.95] = 1
markers[data > 0.95] = 2

# Run random walker algorithm
labels = random_walker(data, markers, beta=10, mode='bf')

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
                                    sharex=True, sharey=True)
ax1.imshow(data, cmap='gray', interpolation='nearest')
ax1.axis('off')
ax1.set_title('Noisy data')
ax2.imshow(markers, cmap='magma', interpolation='nearest')
ax2.axis('off')
ax2.set_title('Markers')
ax3.imshow(labels, cmap='gray', interpolation='nearest')
ax3.axis('off')
ax3.set_title('Segmentation')

fig.tight_layout()
plt.show()
