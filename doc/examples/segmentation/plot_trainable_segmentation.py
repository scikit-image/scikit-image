"""
Trainable segmentation using local features and random forests
==============================================================

A pixel-based segmentation is computed here using local features based on
local intensity, edges and textures at different scales. A user-provided
mask is used to identify different regions. The pixels of the mask are used
to train a random-forest classifier [1]_ from scikit-learn. Unlabeled pixels
are then labeled from the prediction of the classifier.

This segmentation algorithm is called trainable segmentation in other software
such as ilastik [2]_ or ImageJ [3]_ (where it is also called "weka
segmentation").

.. [1] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
.. [2] https://www.ilastik.org/documentation/pixelclassification/pixelclassification
.. [3] https://imagej.net/Trainable_Weka_Segmentation#Training_features_.282D.29
"""


from itertools import combinations_with_replacement
import itertools
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, feature, segmentation
from skimage import img_as_float32
from sklearn.ensemble import RandomForestClassifier


def _features_sigma(img, sigma, intensity=True, edges=True, texture=True):
    """Features for a single value of the Gaussian blurring parameter ``sigma``
    """
    features = []
    img_blur = filters.gaussian(img, sigma)
    if intensity:
        features.append(img_blur)
    if edges:
        features.append(filters.sobel(img_blur))
    if texture:
        H_elems = [np.gradient(np.gradient(img_blur)[ax0], axis=ax1)
            for ax0, ax1 in combinations_with_replacement(range(img.ndim), 2)]
        eigvals = feature.hessian_matrix_eigvals(H_elems)
        for eigval_mat in eigvals:
            features.append(eigval_mat)
    return features


def _compute_features_gray(img, intensity=True, edges=True, texture=True,
                          sigma_min=0.5, sigma_max=16):
    """Features for a single channel image. ``img`` can be 2d or 3d.
    """
    # computations are faster as float32
    img = img_as_float32(img)
    sigmas = np.logspace(np.log2(sigma_min), np.log2(sigma_max),
            num=int(np.log2(sigma_max) - np.log2(sigma_min) + 1), base=2, endpoint=True)
    n_sigmas = len(sigmas)
    all_results = [_features_sigma(img, sigma, intensity=intensity, edges=edges, texture=texture) for sigma in sigmas]
    return list(itertools.chain.from_iterable(all_results))


def compute_features(img, multichannel=True,
                          intensity=True, edges=True, texture=True,
                          sigma_min=0.5, sigma_max=16):
    """Features for a single- or multi-channel image.
    """
    if img.ndim == 3 and multichannel:
        all_results = (_compute_features_gray(
                img[..., dim], intensity=intensity, edges=edges, texture=texture,
                sigma_min=sigma_min, sigma_max=sigma_max) for dim in range(img.shape[-1]))
        features = list(itertools.chain.from_iterable(all_results))
    else:
        features = _compute_features_gray(
            img[..., dim], intensity=intensity, edges=edges, texture=texture,
            sigma_min=sigma_min, sigma_max=sigma_max)
    return np.array(features)


def trainable_segmentation(img, mask, multichannel=True,
                           intensity=True, edges=True, texture=True,
                           sigma_min=0.5, sigma_max=16, downsample=10):
    """
    Segmentation using labeled parts of the image and a random forest classifier.
    """
    print(img.shape)
    t1 = time()
    features = compute_features(im, multichannel=multichannel,
                            intensity=intensity, edges=edges, texture=texture,
                            sigma_min=sigma_min, sigma_max=sigma_max)
    t2 = time()
    training_data = features[:, mask > 0].T
    training_labels = mask[mask>0].ravel()
    data = features[:, mask == 0].T
    t3 = time()
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(training_data[::downsample], training_labels[::downsample])
    t4 = time()
    labels = clf.predict(data)
    t5 = time()
    result = np.copy(mask)
    result[mask == 0] = labels
    print("compute features", t2 - t1)
    print("fit", t4 - t3)
    print("predict", t5 - t4)
    return result, clf


from time import time
# This image is in the public domain and could be added to the Gitlab data repo
filename = 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Normal_Epidermis_and_Dermis_with_Intradermal_Nevus_10x.JPG/1280px-Normal_Epidermis_and_Dermis_with_Intradermal_Nevus_10x.JPG'
im = io.imread(filename)

# Build a mask for training the segmentation.
# Here we use rectangles but visualization libraries such as plotly
# (and napari?) can be used to draw a mask on the image.
mask = np.zeros(im.shape[:2], dtype=np.uint8)
mask[:100] = 1
mask[:170, :400] = 1
mask[600:900, 200:650] = 2
mask[330:430, 210:320] = 3
mask[260:340, 60:170] = 4
mask[150:200, 720:860] = 4

sigma_min = 1
sigma_max = 32
result, clf = trainable_segmentation(im, mask, multichannel=True,
                        intensity=True, edges=False, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max, downsample=15)

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
ax[0].imshow(segmentation.mark_boundaries(im, result, mode='thick'))
ax[0].contour(mask)
ax[0].set_title('Image, mask and segmentation boundaries')
ax[1].imshow(result)
ax[1].set_title('Segmentation')
fig.tight_layout()

##############################################################################
# Feature importance
# ------------------
#
# We inspect below the importance of the different features, as computed by
# scikit-learn. Intensity features have a much higher importance than texture
# features. It can be tempting to use this information to reduce the number of
# features given to the classifier, in order to reduce the computing time.
# However, this can lead to overfitting and a degraded result at the boundary
# between regions.

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
l = len(clf.feature_importances_)
feature_importance = (
        clf.feature_importances_[:l//3],
        clf.feature_importances_[l//3:2*l//3],
        clf.feature_importances_[2*l//3:])
sigmas = np.logspace(np.log2(sigma_min), np.log2(sigma_max),
            num=int(np.log2(sigma_max) - np.log2(sigma_min) + 1),
            base=2, endpoint=True)
for ch, color in zip(range(3), ['r', 'g', 'b']):
    ax[0].plot(sigmas, feature_importance[ch][::3], 'o', color=color)
    ax[0].set_title("Intensity features")
    ax[0].set_xlabel("$\sigma$")
for ch, color in zip(range(3), ['r', 'g', 'b']):
    ax[1].plot(sigmas, feature_importance[ch][1::3], 'o', color=color)
    ax[1].plot(sigmas, feature_importance[ch][2::3], 's', color=color)
    ax[1].set_title("Texture features")
    ax[1].set_xlabel("$\sigma$")

fig.tight_layout()
plt.show()
