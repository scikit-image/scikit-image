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

from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier

from skimage.data import cbcl_database
from skimage.transform import integral_image
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


def extract_feature_image(img, feature_type):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type)


images = cbcl_database()

feature_types = ['type-2-x', 'type-2-y']

X = np.array(Parallel(n_jobs=-1)(
    delayed(extract_feature_image)(img, feature_types)
    for img in images))
y = np.array([1] * 100 + [0] * 100)

feature_coord, feature_type = haar_like_feature_coord(images.shape[2],
                                                      images.shape[1],
                                                      feature_types)

clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                             max_features=100,
                             n_jobs=-1, random_state=0)
clf.fit(X, y)

idx_sorted = np.argsort(clf.feature_importances_)[::-1]

fig, axs = plt.subplots(3, 2)
for idx, ax in enumerate(axs.ravel()):
    image = images[0]
    image = draw_haar_like_feature(image, 0, 0,
                                   images.shape[2],
                                   images.shape[1],
                                   [feature_coord[idx_sorted[idx]]])
    ax.imshow(image)

fig.suptitle('The most important features')
plt.tight_layout()
plt.show()

# image = rgb2gray(astronaut())
# plt.figure()
# coord, _ = haar_like_feature_coord(50, 50, 'type-4')
# plt.imshow(draw_haar_like_feature(image, 125, 200, 50, 50,
#                                   coord, max_n_features=3,
#                                   random_state=0))
# plt.plot([200, 200 + 50, 200 + 50, 200, 200],
#          [125, 125, 125 + 50, 125 + 50, 125],
#          label='ROI')
# plt.axis([100, 300, 100, 300])
# plt.gca().invert_yaxis()
# plt.axis('off')
# plt.title('Example of few Haar-like feature extracted in ROI area')
# plt.legend()

# ###############################################################################
# # The descriptor
# ###############################################################################
# # As previously mentioned, the actual descriptor corresponds to the subtraction
# # between the rectangle from the integral of the original image. The
# # :func:`haar_like_feature` allows to extract those features for a specific ROI

# haar_features = haar_like_feature(image, 125, 200, 24, 24)
# print('{} Haar-like features have been extracted: {}'.format(
#     haar_features.shape[0], haar_features))


# plt.show()
