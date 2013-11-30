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
                             corner_peaks, ORB)
from skimage.color import rgb2gray
from skimage import img_as_float
import matplotlib.pyplot as plt


img1_color = data.lena()
img2_color = tf.rotate(img1_color, 180)
tform = tf.AffineTransform(scale=(1.3, 1.1), rotation=0.5,
                           translation=(0, -200))
img3_color = tf.warp(img1_color, tform)
img1 = rgb2gray(img1_color)
img2 = rgb2gray(img2_color)
img3 = rgb2gray(img3_color)

descriptor_extractor = ORB(n_keypoints=200)
keypoints1, descriptors1 = descriptor_extractor.detect_and_extract(img1)
keypoints2, descriptors2 = descriptor_extractor.detect_and_extract(img2)
keypoints3, descriptors3 = descriptor_extractor.detect_and_extract(img3)

idxs1, idxs2 = match_descriptors(descriptors1, descriptors2, cross_check=True)
src12 = keypoints1[idxs1]
dst12 = keypoints2[idxs2]

idxs1, idxs3 = match_descriptors(descriptors1, descriptors3, cross_check=True)
src13 = keypoints1[idxs1]
dst13 = keypoints3[idxs3]

img12 = np.concatenate((img_as_float(img1_color),
                        img_as_float(img2_color)), axis=1)
img13 = np.concatenate((img_as_float(img1_color),
                        img_as_float(img3_color)), axis=1)

imgs = (img12, img13)
srcs = (src12, src13)
dsts = (dst12, dst13)

offset = img1.shape

fig, ax = plt.subplots(nrows=2, ncols=1)

for i in range(2):

    ax[i].imshow(imgs[i], interpolation='nearest')
    ax[i].axis('off')
    ax[i].axis((0, 2 * offset[1], offset[0], 0))

    src = srcs[i]
    dst = dsts[i]

    for m in range(len(src)):
        color = np.random.rand(3, 1)
        ax[i].plot((src[m, 1], dst[m, 1] + offset[1]), (src[m, 0], dst[m, 0]),
                '-', color=color)
        ax[i].scatter(src[m, 1], src[m, 0], facecolors='none', edgecolors=color)
        ax[i].scatter(dst[m, 1] + offset[1], dst[m, 0], facecolors='none',
                   edgecolors=color)

plt.show()
