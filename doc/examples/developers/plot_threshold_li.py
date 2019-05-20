"""
===============
Li thresholding
===============

In 1993, Li and Lee proposed a new criterion for finding the "optimal"
threshold to distinguish between the background and foreground of an image
[1]_. They proposed that minimizing the *cross-entropy* between the foreground
and the foreground mean, and the background and the background mean, would give
the best threshold in most situations.

Until 1998, though, the way to find this threshold was by trying all possible
thresholds and then choosing the one with the smallest cross-entropy. At that
point, Li and Tam implemented a new, iterative method to more quickly find the
optimum point by using the slope of the cross-entropy [2]_. This is the method
implemented in scikit-image's :func:`skimage.filters.threshold_li`.

Here, we demonstrate the cross-entropy and its optimization by Li's iterative
method.

.. [1] Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"
       Pattern Recognition, 26(4): 617-625
       :DOI:`10.1016/0031-3203(93)90115-D`
.. [2] Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum
       Cross Entropy Thresholding" Pattern Recognition Letters, 18(8): 771-776
       :DOI:`10.1016/S0167-8655(98)00057-9`
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import filters
from skimage.filters.thresholding import _cross_entropy

cell = data.cell()
camera = data.camera()

###############################################################################
# First, we let's plot the cross entropy for the :func:`skimage.data.camera`
# image at all possible thresholds.

thresholds = np.arange(np.min(camera) + 1.5, np.max(camera) - 1.5)
entropies = [_cross_entropy(camera, t) for t in thresholds]

optimal_camera_threshold = thresholds[np.argmin(entropies)]

fig, ax = plt.subplots(1, 3, figsize=(8, 3))

ax[0].imshow(camera, cmap='gray')
ax[0].set_title('image')
ax[0].set_axis_off()

ax[1].imshow(camera > optimal_camera_threshold, cmap='gray')
ax[1].set_title('thresholded')
ax[1].set_axis_off()

ax[2].plot(thresholds, entropies)
ax[2].set_xlabel('thresholds')
ax[2].set_ylabel('cross-entropy')
ax[2].vlines(optimal_camera_threshold,
             ymin=np.min(entropies) - 0.05 * np.ptp(entropies),
             ymax=np.max(entropies) - 0.05 * np.ptp(entropies))
ax[2].set_title('optimal threshold')

fig.tight_layout()

print('The brute force optimal threshold is:', optimal_camera_threshold)
print('The computed optimal threshold is:', filters.threshold_li(camera))

plt.show()

###############################################################################
# Next, let's use the ``iter_callback`` feature of ``threshold_li`` to examine
# the optimization process as it happens.

iter_thresholds = []

optimal_threshold = filters.threshold_li(camera,
                                         iter_callback=iter_thresholds.append)
iter_entropies = [_cross_entropy(camera, t) for t in iter_thresholds]

print('Only', len(iter_thresholds), 'examined.')

fig, ax = plt.subplots()

ax.plot(thresholds, entropies, label='all threshold entropies')
ax.plot(iter_thresholds, iter_entropies, label='optimization path')
ax.scatter(iter_thresholds, iter_entropies, c='C1')
ax.legend(loc='upper right')

plt.show()
