"""
=======================
BRIEF binary descriptor
=======================

This example demonstrates the BRIEF binary description algorithm.

The descriptor consists of relatively few bits and can be computed using
a set of intensity difference tests. The short binary descriptor results
in low memory footprint and very efficient matching based on the Hamming
distance metric.

However, BRIEF does not provide rotation-invariance and scale scale-invariance
can be achieved by detecting and extracting features at different scales.

The ORB feature detection and binary description algorithm is an extension to
the BRIEF method and provides rotation and scale-invariance, see
`skimage.feature.ORB`.

"""
import numpy as np
from skimage import data
from skimage import transform as tf
from skimage.feature import (match_descriptors, corner_peaks, corner_harris,
                             plot_matches, BRIEF)
from skimage.color import rgb2gray
from skimage import img_as_float
import matplotlib.pyplot as plt


img1 = rgb2gray(data.lena())
tform = tf.AffineTransform(scale=(1.2, 1.2), translation=(0, -100))
img2 = tf.warp(img1, tform)
img3 = tf.rotate(img1, 25)

keypoints1 = corner_peaks(corner_harris(img1), min_distance=5)
keypoints2 = corner_peaks(corner_harris(img2), min_distance=5)
keypoints3 = corner_peaks(corner_harris(img3), min_distance=5)

extractor = BRIEF()

descriptors1, mask1 = extractor.extract(img1, keypoints1)
descriptors2, mask2 = extractor.extract(img2, keypoints2)
descriptors3, mask3 = extractor.extract(img3, keypoints3)

keypoints1 = keypoints1[mask1]
keypoints2 = keypoints2[mask2]
keypoints3 = keypoints3[mask3]

fig, ax = plt.subplots(nrows=2, ncols=1)

plt.gray()

idxs1, idxs2 = match_descriptors(descriptors1, descriptors2, cross_check=True)
plot_matches(ax[0], img1, img2, keypoints1, keypoints2, idxs1, idxs2)
ax[0].axis('off')

idxs1, idxs3 = match_descriptors(descriptors1, descriptors3, cross_check=True)
plot_matches(ax[1], img1, img3, keypoints1, keypoints3, idxs1, idxs3)
ax[1].axis('off')

plt.show()
