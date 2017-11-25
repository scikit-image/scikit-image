"""
============================
Haar-like feature descriptor
============================

Haar-like feature descriptors are a set of simple features which was used in
the first real-time face detector [1]_. These features can be efficiently
computed on any scale in constant time, using an integral image [1]_ and by
subtracting rectangle area. The following example will show the mechanism to
build this family of descriptors.

References
----------

.. [1] Viola, Paul, and Michael J. Jones. "Robust real-time face
       detection." International journal of computer vision 57.2
       (2004): 137-154.
       http://www.merl.com/publications/docs/TR2004-043.pdf
       DOI: 10.1109/CVPR.2001.990517

"""
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature

print(__doc__)

###############################################################################
# The different Haar-like feature descriptors
###############################################################################
# The Haar-like feature descriptors come into 5 different types as illustrated
# in the figure below. The value of the descriptor consists in the subtraction
# between the green and red rectangles.

images = [np.zeros((2, 2)), np.zeros((2, 2)),
          np.zeros((3, 3)), np.zeros((3, 3)),
          np.zeros((2, 2))]

feature_types = ['type-2-x', 'type-2-y',
                 'type-3-x', 'type-3-y',
                 'type-4']

fig, axs = plt.subplots(3, 2)
for ax, img, feat_t in zip(np.ravel(axs), images, feature_types):
    coord, _ = haar_like_feature_coord(img.shape[0], img.shape[1], feat_t)
    haar_feature = draw_haar_like_feature(img, 0, 0,
                                          img.shape[0],
                                          img.shape[1],
                                          coord,
                                          max_n_features=1,
                                          random_state=0)
    ax.imshow(haar_feature)
    ax.set_title(feat_t)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle('The different Haar-like feature descriptors')
plt.axis('off')
plt.show()

###############################################################################
# The feature scalar value corresponds to the sum of the pixel intensities of
# the red area is subtracted to the sum of the pixel intensities of the green
# area. In practice, the haar-like features will be placed in all possible
# location of an image and a feature value will be extracted for each of these
# locations.
