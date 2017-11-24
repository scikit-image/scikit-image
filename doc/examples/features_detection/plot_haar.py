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
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals.joblib import Parallel, delayed

from skimage.data import cbcl_face_database
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
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle('The different Haar-like feature descriptors')
plt.axis('off')

###############################################################################
# The usual feature extraction scheme
###############################################################################
# The procedure to extract the Haar-like feature for an image is quite easy: a
# region of interest (ROI) is defined for which all possible feature will be
# extracted. The integral image of this ROI will be computed and all possible
# features will be computed.


def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)


# trick such that we can pickle this function when building the doc with
# sphinx-gallery
sys.modules['__main__'].extract_feature_image = extract_feature_image


###############################################################################
# We will use a subset of the CBCL which is composed of 100 face images and 100
# non-face images. Each image has been resized to a ROI of 19 by 19 pixels. We
# will keep 75 images from each group to train a classifier and check which
# extracted features are the most salient.

images = cbcl_face_database()
# For a gain of speed, only the two first types of features will be extracted.
feature_types = ['type-2-x', 'type-2-y']

X = np.array(Parallel(n_jobs=-1)(
    delayed(extract_feature_image)(img, feature_types)
    for img in images))
y = np.array([1] * 100 + [0] * 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=150,
                                                    random_state=0)

# Extract all possible features to be able to select the salient one.
feature_coord, feature_type = haar_like_feature_coord(images.shape[2],
                                                      images.shape[1],
                                                      feature_types)

# Train a random forest classifier and check the feature importance
clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                             max_features=100, n_jobs=-1, random_state=0)
clf.fit(X_train, y_train)

idx_sorted = np.argsort(clf.feature_importances_)[::-1]

###############################################################################
# A random forest classifier can be trained in order to select the most salient
# features, specifically for face classification. The idea is to check which
# features are the most often used by the ensemble of trees. Below, we are
# plotting the six most salient features found by the random forest.

fig, axs = plt.subplots(3, 2)
for idx, ax in enumerate(axs.ravel()):
    image = images[0]
    image = draw_haar_like_feature(image, 0, 0,
                                   images.shape[2],
                                   images.shape[1],
                                   [feature_coord[idx_sorted[idx]]])
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle('The most important features')

###############################################################################
# We can select the most important features by checking the cumulative sum of
# the feature importance index; we kept features representing 70% of the
# cumulative value which represent only 3% of the total number of features.

cdf_feature_importances = np.cumsum(clf.feature_importances_[idx_sorted[::-1]])
cdf_feature_importances /= np.max(cdf_feature_importances)
significant_feature = np.count_nonzero(cdf_feature_importances > 0.3)
print('There is {} features which are considered important.'.format(
    significant_feature))

# Select the most informative features
selected_feature_coord = feature_coord[idx_sorted[:significant_feature]]
selected_feature_type = feature_type[idx_sorted[:significant_feature]]
# Note: we could select those features from the
# original matrix X but we would like to emphasize the usage of `feature_coord`
# and `feature_type` to recompute a subset of desired features.
X = np.array(Parallel(n_jobs=-1)(
    delayed(extract_feature_image)(img, selected_feature_type,
                                   selected_feature_coord)
    for img in images))
y = np.array([1] * 100 + [0] * 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=150,
                                                    random_state=0)

###############################################################################
# Once the feature are extracted, we can train and test the a new classifier.

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
plt.show()
