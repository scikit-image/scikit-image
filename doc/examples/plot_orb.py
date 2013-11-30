"""
==========================================
ORB feature detector and binary descriptor
==========================================

This example demonstrates the ORB feature detection and binary description
algorithm. It uses an oriented FAST detection method and the rotated BRIEF
descriptors.

ORB is comparatively scale- and rotation-invariant. As a binary descriptor it
allows to employ the very efficient Hamming distance metric for matching and
is thus preferred for real-time applications.

"""
import numpy as np
from skimage import data
from skimage import transform as tf
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from skimage.color import rgb2gray
from skimage import img_as_float
import matplotlib.pyplot as plt


img1 = rgb2gray(data.lena())
img2 = tf.rotate(img1, 180)
tform = tf.AffineTransform(scale=(1.3, 1.1), rotation=0.5,
                           translation=(0, -200))
img3 = tf.warp(img1, tform)

descriptor_extractor = ORB(n_keypoints=200)
keypoints1, descriptors1 = descriptor_extractor.detect_and_extract(img1)
keypoints2, descriptors2 = descriptor_extractor.detect_and_extract(img2)
keypoints3, descriptors3 = descriptor_extractor.detect_and_extract(img3)

fig, ax = plt.subplots(nrows=2, ncols=1)

plt.gray()

idxs1, idxs2 = match_descriptors(descriptors1, descriptors2, cross_check=True)
plot_matches(ax[0], img1, img2, keypoints1, keypoints2,
             idxs1, idxs2)
ax[0].axis('off')

idxs1, idxs3 = match_descriptors(descriptors1, descriptors3, cross_check=True)
plot_matches(ax[1], img1, img3, keypoints1, keypoints3,
             idxs1, idxs3)
ax[1].axis('off')

plt.show()
