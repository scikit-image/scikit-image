"""
============================
Haar-like feature descriptor
============================

Haar-like feature descriptors are a set of simple features which was used in
the first real-time face detector [1]_. Indeed, these features can be
efficiently computed using a so-called integral image and by subtracting
rectangle area. The following example will show the mechanism to build this
family of descriptors.

References
----------

.. [1] Viola, Paul, and Michael J. Jones. "Robust real-time face
       detection." International journal of computer vision 57.2
       (2004): 137-154.
       http://www.merl.com/publications/docs/TR2004-043.pdf
       DOI: 10.1109/CVPR.2001.990517

"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.data import astronaut
from skimage.color import rgb2gray

from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature

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

feature = [draw_haar_like_feature(img, 0, 0, img.shape[1], img.shape[0],
                                  feat_t, max_n_features=1)
           for img, feat_t in zip(images, feature_types)]

fig, axs = plt.subplots(3, 2)
for ax, img, feat_t in zip(np.ravel(axs), images, feature_types):
    ax.imshow(draw_haar_like_feature(img, 0, 0, img.shape[1], img.shape[0],
                                     feat_t, max_n_features=1,
                                     random_state=0))
    ax.set_title(feat_t)
fig.suptitle('The different Haar-like feature descriptors')
plt.axis('off')
plt.tight_layout()

###############################################################################
# The usual feature extraction scheme
###############################################################################
# The procedure to extract the Haar-like feature for an image is quite easy:
# a region of interest (ROI) is defined for which all possible feature will
# be extracted. In order to compute those features, an integral image needs to
# be provided. Typically, for a ROI of 24 by 24 pixels, the total amount of
# features which can be extracted is 160,000.

image = rgb2gray(astronaut())
plt.figure()
plt.imshow(draw_haar_like_feature(image, 125, 200, 50, 50,
                                  'type-4', max_n_features=3,
                                  random_state=0))
plt.plot([200, 200 + 50, 200 + 50, 200, 200],
         [125, 125, 125 + 50, 125 + 50, 125],
         label='ROI')
plt.axis([100, 300, 100, 300])
plt.gca().invert_yaxis()
plt.axis('off')
plt.title('Example of few Haar-like feature extracted in ROI area')
plt.legend()

###############################################################################
# The descriptor
###############################################################################
# As previously mentioned, the actual descriptor corresponds to the subtraction
# between the rectangle from the integral of the original image. The
# :func:`haar_like_feature` allows to extract those features for a specific ROI

haar_features = haar_like_feature(image, 125, 200, 24, 24)
print('{} Haar-like features have been extracted: {}'.format(
    haar_features.shape[0], haar_features))


plt.show()
